"""
preprocess.py -- Build a dual-target RecBole-CDR dataset pair from two raw
interaction CSVs, following Section 4.1 of the DGCDR paper.

Sampling procedure (paper, Sec. 4.1):
    Step 1. Extract overlapping users from both domains;
    Step 2. Filter users: remove those with fewer than N interactions;
    Step 3. Filter items: remove those with fewer than N interactions;
    Step 4. Iterate Steps 2-3 until all users and items have >= N interactions;
    Step 5. Re-extract overlapping users in both domains after filtering.

Step 5 is applied ONCE, after Steps 2-4 have converged. Dropping the users that
survived in only one domain removes interactions from items, so the item-side
N-core no longer holds in the final files. That is what the released datasets
actually look like: their per-domain minimum item degree is 1 while the minimum
user degree is exactly N, and the counts reproduce Table 2 of the paper
(e.g. AmazonElec 35,827 users / 62,548 items / 811,969 inters).
Pass --strict_core to instead iterate Steps 2-5 to a joint fixed point.

Output layout (matches the repository conventions):
    dataset/<SRC>_<TGT>_commonUser_<N>-core/<SRC>_<TGT>_commonUser_<N>-core.inter
    dataset/<TGT>_<SRC>_commonUser_<N>-core/<TGT>_<SRC>_commonUser_<N>-core.inter
    recbole_cdr/properties/dataset/<SRC>_<TGT>_commonUser_<N>-core.yaml
    recbole_cdr/properties/dataset/<TGT>_<SRC>_commonUser_<N>-core.yaml

Example:
    python preprocess.py --source_csv csv_datasets/google_hotels.csv \
                         --target_csv csv_datasets/google_restaurants.csv \
                         --source_name GoogleHotels --target_name GoogleRestaurants \
                         --k_core 5
"""

import argparse
import os
import time
from string import Template

import pandas as pd

# === DEFAULT CONFIGURATION ===
DEFAULT_SOURCE_CSV = 'new_datasets\Beauty_and_Personal_Care.csv'
DEFAULT_TARGET_CSV = 'new_datasets\Health_and_Household.csv'
DEFAULT_SOURCE_NAME = 'Beauty_and_Personal_Care'
DEFAULT_TARGET_NAME = 'Health_and_Household'
DEFAULT_K_CORE = 5
DEFAULT_OUT_DIR = 'dataset'
DEFAULT_CONFIG_DIR = os.path.join('recbole_cdr', 'properties', 'dataset')
# =============================

# Canonical field -> accepted source column names (matched case-insensitively).
COLUMN_ALIASES = {
    'user_id': ['user_id', 'reviewerID', 'reviewer_id', 'userID', 'uid'],
    'item_id': ['item_id', 'parent_asin', 'asin', 'itemID', 'gmap_id', 'business_id', 'iid'],
    'rating': ['rating', 'overall', 'stars', 'score'],
    'timestamp': ['timestamp', 'unixReviewTime', 'unix_time', 'time', 'date'],
}
REQUIRED_FIELDS = ('user_id', 'item_id', 'rating')
FIELD_TYPES = {'user_id': 'token', 'item_id': 'token', 'rating': 'float', 'timestamp': 'float'}

# `timestamp` is deliberately absent from load_col: the README documents the
# .inter format as (user_id, item_id, rating), every shipped dataset yaml loads
# only those three fields, and the split uses `order: RO`. The column is written
# to the file only for provenance, exactly like the shipped Douban files do.
YAML_TEMPLATE = Template("""seed: $seed
field_separator: "\\t"
#save_dataloaders: True
#save_dataset: True
embedding_size: 64
learning_rate: 0.001


eval_args:
#  split: { 'LS': 'valid_and_test' }  #leave-one-out data splitting
  split: {'RS':[0.6,0.2,0.2]}
  split_valid: {'RS': [0.8,0.2]}  # The source domain is split by 8:2 for training and validation.
  group_by: user
  order: RO
  mode: full
repeatable: True
metrics: [ "MRR", "Recall","NDCG","Hit"]
topk: [20]
valid_metric: Recall@20

# Training settings
epochs: 400
train_batch_size: $train_batch_size
eval_batch_size: $eval_batch_size

#train_neg_sample_args: None

source_domain:
  dataset: $src_dataset
  data_path: '$data_path'
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  threshold:
    rating: $rating_threshold                    # (dict) 0/1 labels will be generated according to the pairs.
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[0,inf)"
  item_inter_num_interval: "[0,inf)"
  val_interval:
    rating: "[0,inf)"
  drop_filter_field: True


target_domain:
  dataset: $tgt_dataset
  data_path: '$data_path'
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  threshold:
    rating: $rating_threshold                    # (dict) 0/1 labels will be generated according to the pairs.
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[0,inf)"
  item_inter_num_interval: "[0,inf)"
  val_interval:
    rating: "[0,inf)"
  drop_filter_field: True
""")


def resolve_columns(header, overrides):
    """Maps the CSV header onto the canonical field names."""
    by_lower = {str(c).lower(): c for c in header}
    resolved = {}
    for field, aliases in COLUMN_ALIASES.items():
        forced = overrides.get(field)
        if forced:
            if forced not in header:
                raise SystemExit(f"ERROR: column '{forced}' not found. Header: {list(header)}")
            resolved[field] = forced
            continue
        for alias in aliases:
            if alias.lower() in by_lower:
                resolved[field] = by_lower[alias.lower()]
                break
    return resolved


def to_epoch_seconds(series):
    """Best-effort conversion of a timestamp column to numeric epoch seconds."""
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().mean() > 0.5:
        return numeric
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    if parsed.notna().mean() > 0.5:
        return parsed.astype('int64') // 10 ** 9
    return None


def load_domain(path, label, overrides):
    """Loads a domain CSV, normalising column names and dropping invalid rows."""
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: CSV not found for {label}: {path}")

    header = list(pd.read_csv(path, nrows=0).columns)
    cols = resolve_columns(header, overrides)
    missing = [f for f in REQUIRED_FIELDS if f not in cols]
    if missing:
        raise SystemExit(
            f"ERROR: {label} is missing required column(s) {missing}.\n"
            f"       Header: {header}\n"
            f"       Map them explicitly with --user_col / --item_col / --rating_col / --time_col."
        )

    # Reading only the needed columns keeps the memory footprint down on the
    # large Amazon dumps.
    df = pd.read_csv(
        path,
        usecols=list(cols.values()),
        dtype={cols['user_id']: str, cols['item_id']: str},
    )
    df = df.rename(columns={src: field for field, src in cols.items()})

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    if 'timestamp' in df.columns:
        converted = to_epoch_seconds(df['timestamp'])
        if converted is None:
            print(f"      [{label}] timestamp column is not parseable, dropping it")
            df = df.drop(columns=['timestamp'])
        else:
            df['timestamp'] = converted

    before = len(df)
    df = df.dropna(subset=[c for c in ('user_id', 'item_id', 'rating', 'timestamp') if c in df.columns])
    if before != len(df):
        print(f"      [{label}] dropped {before - len(df)} rows with missing values")

    # Ids end up in a tab-separated atomic file; a stray tab would corrupt it.
    for field in ('user_id', 'item_id'):
        df[field] = df[field].str.strip()
        bad = int(df[field].str.contains('\t', regex=False).sum())
        if bad:
            raise SystemExit(f"ERROR: {label} has {bad} '{field}' values containing a tab character.")
    df = df[(df['user_id'] != '') & (df['item_id'] != '')]

    return df.reset_index(drop=True)


def deduplicate(df, label):
    """Collapses repeated (user, item) pairs, keeping the most recent one.

    Raw Amazon reviews map several variant ASINs onto a single `parent_asin`, so
    the same user can appear more than once on the same item. Duplicates would
    inflate the N-core degrees and, under the random 60/20/20 split, place the
    very same pair in both the training and the test set.
    """
    before = len(df)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', kind='mergesort')
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    removed = before - len(df)
    if removed:
        print(f"      [{label}] removed {removed} duplicate (user, item) pairs")
    return df.sort_index().reset_index(drop=True)


def k_core_filter(df, k, label):
    """Steps 2-4: iterate the user/item filtering until both reach the N-core."""
    iteration = 0
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        # Step 2: keep users with at least k interactions
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]

        # Step 3: keep items with at least k interactions
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]

        print(f"      [{label} | iter {iteration}] {len(df)} interactions remaining")
        if df.empty:
            break
    return df


def describe(df, label):
    """Prints Table 2-style statistics for one domain."""
    users, items, inters = df['user_id'].nunique(), df['item_id'].nunique(), len(df)
    if not inters:
        print(f"      {label:<28} EMPTY")
        return
    sparsity = 100.0 * (1.0 - inters / (users * items))
    min_user = int(df['user_id'].value_counts().min())
    min_item = int(df['item_id'].value_counts().min())
    print(f"      {label:<28} users={users:<8} items={items:<8} inters={inters:<10} "
          f"sparsity={sparsity:.4f}%  min_user_deg={min_user}  min_item_deg={min_item}")


def write_inter(df, directory, dataset_name, keep_timestamp):
    os.makedirs(directory, exist_ok=True)
    fields = ['user_id', 'item_id', 'rating']
    if keep_timestamp and 'timestamp' in df.columns:
        fields.append('timestamp')
    path = os.path.join(directory, f"{dataset_name}.inter")
    df = df[fields].astype({f: float for f in fields if FIELD_TYPES[f] == 'float'})
    df.to_csv(
        path, sep='\t', index=False,
        header=[f"{f}:{FIELD_TYPES[f]}" for f in fields],
    )
    return path


def write_yaml(path, src_dataset, tgt_dataset, args):
    data_path = './' + args.out_dir.replace('\\', '/').rstrip('/') + '/'
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(YAML_TEMPLATE.substitute(
            seed=args.seed,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            rating_threshold=args.rating_threshold,
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            data_path=data_path,
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Builds a RecBole-CDR dual-target dataset pair following Sec. 4.1 of the DGCDR paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--source_csv', type=str, default=DEFAULT_SOURCE_CSV, help='Path to the source domain CSV')
    parser.add_argument('--target_csv', type=str, default=DEFAULT_TARGET_CSV, help='Path to the target domain CSV')
    parser.add_argument('--source_name', type=str, default=DEFAULT_SOURCE_NAME, help='Short name for the source domain')
    parser.add_argument('--target_name', type=str, default=DEFAULT_TARGET_NAME, help='Short name for the target domain')
    parser.add_argument('--k_core', type=int, default=DEFAULT_K_CORE, help='N for the N-core filtering')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR, help='Base output directory for the .inter files')
    parser.add_argument('--config_dir', type=str, default=DEFAULT_CONFIG_DIR, help='Directory for the generated .yaml')

    parser.add_argument('--user_col', type=str, default=None, help='Override the user id column name')
    parser.add_argument('--item_col', type=str, default=None, help='Override the item id column name')
    parser.add_argument('--rating_col', type=str, default=None, help='Override the rating column name')
    parser.add_argument('--time_col', type=str, default=None, help='Override the timestamp column name')

    parser.add_argument('--no_dedup', action='store_true', help='Keep repeated (user, item) pairs')
    parser.add_argument('--strict_core', action='store_true',
                        help='Iterate Steps 2-5 to a fixed point so the N-core still holds after the final '
                             'overlap extraction (deviates from the released datasets)')
    parser.add_argument('--keep_timestamp', action='store_true',
                        help='Also write a timestamp column into the .inter files (not loaded by RecBole)')
    parser.add_argument('--min_common_users', type=int, default=100,
                        help='Abort if fewer common users survive the filtering')

    parser.add_argument('--seed', type=int, default=2022, help='seed written into the generated yaml')
    parser.add_argument('--rating_threshold', type=int, default=4, help='Positive-label threshold in the yaml')
    parser.add_argument('--train_batch_size', type=int, default=4096, help='train_batch_size in the yaml')
    parser.add_argument('--eval_batch_size', type=int, default=40960, help='eval_batch_size in the yaml')
    args = parser.parse_args()

    if args.source_name == args.target_name:
        raise SystemExit("ERROR: --source_name and --target_name must differ.")
    if args.k_core < 1:
        raise SystemExit("ERROR: --k_core must be >= 1.")

    start_time = time.time()
    overrides = {'user_id': args.user_col, 'item_id': args.item_col,
                 'rating': args.rating_col, 'timestamp': args.time_col}

    print("[1/6] Loading CSVs...")
    source_df = load_domain(args.source_csv, args.source_name, overrides)
    target_df = load_domain(args.target_csv, args.target_name, overrides)
    print(f"      {args.source_name}: {len(source_df)} interactions (original total)")
    print(f"      {args.target_name}: {len(target_df)} interactions (original total)")

    print("\n[2/6] Deduplicating (user, item) pairs...")
    if args.no_dedup:
        print("      skipped (--no_dedup)")
    else:
        source_df = deduplicate(source_df, args.source_name)
        target_df = deduplicate(target_df, args.target_name)

    print("\n[3/6] Extracting common users (Step 1 from the paper)...")
    common_users = set(source_df['user_id']) & set(target_df['user_id'])
    source_df = source_df[source_df['user_id'].isin(common_users)]
    target_df = target_df[target_df['user_id'].isin(common_users)]
    print(f"      Common users found: {len(common_users)}")
    if not common_users:
        raise SystemExit("ERROR: the two domains share no user. Check that the user ids live in the same id space.")

    print(f"\n[4/6] Iterative {args.k_core}-core filtering + overlap re-extraction (Steps 2-5 from the paper)...")
    round_no = 0
    while True:
        round_no += 1
        source_df = k_core_filter(source_df, args.k_core, args.source_name)
        target_df = k_core_filter(target_df, args.k_core, args.target_name)

        # Step 5: re-extract the overlapping users after filtering.
        common_final = set(source_df['user_id']) & set(target_df['user_id'])
        src_next = source_df[source_df['user_id'].isin(common_final)]
        tgt_next = target_df[target_df['user_id'].isin(common_final)]
        converged = len(src_next) == len(source_df) and len(tgt_next) == len(target_df)
        source_df, target_df = src_next, tgt_next
        print(f"      [round {round_no}] common users after Step 5: {len(common_final)}")

        if not args.strict_core or converged or source_df.empty:
            break

    print("\n      === FINAL RESULT ===")
    describe(source_df, args.source_name)
    describe(target_df, args.target_name)
    print(f"      Final common users: {len(common_final)}")
    if not args.strict_core:
        print("      (item degrees below N are expected: Step 5 runs after the N-core, as in the paper's datasets)")

    if len(common_final) < max(args.min_common_users, 1) or source_df.empty or target_df.empty:
        raise SystemExit(
            f"ERROR: only {len(common_final)} common users survived, below --min_common_users="
            f"{args.min_common_users}. Lower --k_core or pick a closer domain pair."
        )

    # RecBole binarises the ratings with `threshold: rating: N`; users left
    # without any positive interaction contribute nothing to a `group_by: user`
    # split and will drag every ranking metric down.
    for df, name in ((source_df, args.source_name), (target_df, args.target_name)):
        positive = df[df['rating'] >= args.rating_threshold]
        without = len(common_final) - positive['user_id'].nunique()
        share = 100.0 * len(positive) / len(df)
        print(f"      {name}: {share:.1f}% of the interactions are positive "
              f"(rating >= {args.rating_threshold}), {without} users have none")

    print(f"\n[5/6] Saving the .inter files in '{args.out_dir}'...")
    source_dataset_name = f"{args.source_name}_{args.target_name}_commonUser_{args.k_core}-core"
    target_dataset_name = f"{args.target_name}_{args.source_name}_commonUser_{args.k_core}-core"
    source_path = write_inter(source_df, os.path.join(args.out_dir, source_dataset_name),
                              source_dataset_name, args.keep_timestamp)
    target_path = write_inter(target_df, os.path.join(args.out_dir, target_dataset_name),
                              target_dataset_name, args.keep_timestamp)

    print(f"[6/6] Generating the corresponding .yaml files in '{args.config_dir}'...")
    os.makedirs(args.config_dir, exist_ok=True)
    source_yaml_path = os.path.join(args.config_dir, f"{source_dataset_name}.yaml")
    target_yaml_path = os.path.join(args.config_dir, f"{target_dataset_name}.yaml")
    write_yaml(source_yaml_path, source_dataset_name, target_dataset_name, args)
    write_yaml(target_yaml_path, target_dataset_name, source_dataset_name, args)

    print(f"\nFiles successfully created in {time.time() - start_time:.1f} seconds:")
    for path in (source_path, target_path, source_yaml_path, target_yaml_path):
        print(f"  - {path}")
    print(f"\nTo train with {args.target_name} as the target domain, set in run_recbole_cdr.py:")
    print(f"    data_config = '{source_yaml_path.replace(os.sep, '/')}'")
    print("and for the opposite direction:")
    print(f"    data_config = '{target_yaml_path.replace(os.sep, '/')}'")


if __name__ == '__main__':
    main()
