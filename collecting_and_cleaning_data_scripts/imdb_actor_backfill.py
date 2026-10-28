"""Recover only 1985–1990 IMDb feature-film credits for current TMDB leads.

Creates supplements; never rewrites an existing source or model dataset.
"""
from pathlib import Path
import io
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
START_YEAR, END_YEAR = 1985, 1990


def chunks(filename, columns):
    return pd.read_csv(DATA / filename, sep="\t", usecols=columns, dtype="string",
                       na_values=[r"\N"], chunksize=500_000)


def normalize(value):
    return "" if pd.isna(value) else " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def save_snapshot(frame, path):
    text = frame.to_csv(index=False)
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(io.StringIO(text)), check_dtype=False)
    else:
        with path.open("x") as handle:
            handle.write(text)
    print(path.name, frame.shape, flush=True)


def main():
    targets = pd.read_csv(DATA / "cleaned_movies.csv", dtype={"imdb_id": "string"})
    targets["lead_actor"] = targets["top_cast"].astype("string").str.split("|").str[0].str.strip().replace("", pd.NA)
    targets["name_key"] = targets["lead_actor"].map(normalize)
    years = []
    for chunk in chunks("title.basics.tsv.gz", ["tconst", "titleType", "startYear"]):
        years.append(chunk.loc[chunk["titleType"].eq("movie"), ["tconst", "startYear"]])
    years = pd.concat(years, ignore_index=True)
    years["startYear"] = pd.to_numeric(years["startYear"], errors="coerce").astype("Int64")
    years = years.dropna(subset=["startYear"])
    assert years["tconst"].is_unique
    old_years = years.loc[years["startYear"].between(START_YEAR, END_YEAR)]
    old_ids, target_ids = set(old_years["tconst"]), set(targets["imdb_id"].dropna())
    wanted = old_ids | target_ids
    relevant = []
    for number, chunk in enumerate(chunks("title.principals.tsv.gz", ["tconst", "nconst", "category"]), 1):
        mask = chunk["tconst"].isin(wanted) & chunk["category"].isin(["actor", "actress"])
        relevant.append(chunk.loc[mask, ["tconst", "nconst"]])
        if number % 40 == 0:
            print(f"Scanned {number * 500_000:,} principal-credit rows", flush=True)
    relevant = pd.concat(relevant, ignore_index=True).dropna().drop_duplicates()
    current = relevant.loc[relevant["tconst"].isin(target_ids)]
    people = set(current["nconst"])
    names = []
    for chunk in chunks("name.basics.tsv.gz", ["nconst", "primaryName"]):
        names.append(chunk.loc[chunk["nconst"].isin(people)])
    names = pd.concat(names, ignore_index=True)
    names["name_key"] = names["primaryName"].map(normalize)
    current = current.merge(names, on="nconst", validate="many_to_one")
    matched = targets[["tmdb_id", "imdb_id", "name_key"]].merge(
        current, left_on=["imdb_id", "name_key"], right_on=["tconst", "name_key"], how="inner",
    ).drop_duplicates(["tmdb_id", "nconst"])
    matched = matched.loc[matched["name_key"].ne("")]
    counts = matched.groupby("tmdb_id")["nconst"].nunique()
    unique = matched.loc[matched["tmdb_id"].isin(counts[counts.eq(1)].index)]
    target = targets.merge(unique[["tmdb_id", "nconst"]], on="tmdb_id", how="left", validate="one_to_one")
    target["lead_actor_match_status"] = "not_found"
    target.loc[target["nconst"].notna(), "lead_actor_match_status"] = "matched"
    target.loc[target["tmdb_id"].isin(counts[counts.gt(1)].index), "lead_actor_match_status"] = "ambiguous"
    credits = relevant.loc[relevant["tconst"].isin(old_ids) & relevant["nconst"].isin(set(target["nconst"].dropna()))]
    credits = credits.merge(old_years, on="tconst", validate="many_to_one")
    credits = credits.rename(columns={"tconst": "imdb_id", "nconst": "actor_imdb_id"})
    # Existing TMDB-sample credits must not be counted a second time.
    covered_ids = set(pd.read_csv(DATA / "combined_dataset.csv", usecols=["imdb_id"])["imdb_id"].dropna())
    extras = credits.loc[~credits["imdb_id"].isin(covered_ids)]
    histories = {person: group for person, group in extras.groupby("actor_imdb_id")}
    added, needed, unavailable = [], [], []
    for row in target.itertuples():
        requires = max(START_YEAR, row.release_year - 5) <= min(END_YEAR, row.release_year - 1)
        # Avoid mixing disagreeing target release years from the two providers.
        usable = pd.notna(row.nconst) and pd.notna(row.startYear) and row.startYear == row.release_year
        count = 0
        if requires and usable and row.nconst in histories:
            history = histories[row.nconst]
            eligible = history["startYear"].between(row.release_year - 5, row.release_year - 1) & history["imdb_id"].ne(row.imdb_id)
            count = history.loc[eligible, "imdb_id"].nunique()
        added.append(count)
        needed.append(int(requires))
        unavailable.append(int(requires and not usable))
    output = target[["tmdb_id", "imdb_id", "lead_actor", "nconst", "lead_actor_match_status"]].rename(columns={"nconst": "lead_actor_imdb_id"})
    output["imdb_extra_films_5yr"] = added
    output["backfill_needed"] = needed
    output["backfill_unavailable"] = unavailable
    assert output["tmdb_id"].is_unique and output["tmdb_id"].tolist() == targets["tmdb_id"].tolist()
    save_snapshot(credits, DATA / "imdb_lead_actor_1985_1990.csv")
    save_snapshot(output, DATA / "lead_actor_backfill.csv")
    print("Movies receiving extra historical credits:", output["imdb_extra_films_5yr"].gt(0).sum(), flush=True)
    print("Movies in the backfill window:", sum(needed), "; unavailable:", sum(unavailable), flush=True)


if __name__ == "__main__":
    main()
