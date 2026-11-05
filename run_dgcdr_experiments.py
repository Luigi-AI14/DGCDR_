"""Esegue DGCDR sui cinque seed, salvando soltanto log e checkpoint.

Uso: python run_dgcdr_experiments.py
Anteprima senza training o scritture: python run_dgcdr_experiments.py --dry-run
Percorsi VM opzionali: --data-root /path/dataset --output-root /path/experiments

Rilanciare lo stesso comando salta i run completati e riavvia da zero quelli
incompleti. Per cambiare configurazione usare una nuova cartella di output.
Gli originali usano le tabelle in "results & hypara setting"; le estensioni
usano i parametri dei log VM storici, non valori ottimali pubblicati nel paper.
Il protocollo esistente BOTH:400 viene mantenuto, incluso source_split=False.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


# Lascia attiva UNA delle due righe.
GROUP = "original"
# GROUP = "extension"

SEEDS = [2022, 2023, 42, 24, 1]
DATA_ROOT = "dataset"
OUTPUT_ROOT = "experiments"

ROOT = Path(__file__).resolve().parent
PROPERTIES = ROOT / "recbole_cdr" / "properties"
# Base comune per campi, filtri e valutazione; nomi e percorsi sono sovrascritti.
CONFIG_FILES = [
    PROPERTIES / "overall.yaml",
    PROPERTIES / "dataset" / "DoubanMovie_DoubanBook_commonUser_10-core.yaml",
    PROPERTIES / "model" / "DGCDR.yaml",
]

COMMON_HPARAMS = {
    "train_epochs": ["BOTH:400"],
    "learning_rate": 0.0001,
    "embedding_size": 64,
    "mlp_hidden_size": [256],
    "loss_type": "BPR",
    "reg_weight": 0.001,
    "drop_rate": 0.1,
    "n_layers": 3,
    "init_way": "xavier",
    "preference_disentangle": True,
    "connect_way": "concat",
    "fuse_mode": "attention",
}
PARAMETER_NAMES = (
    "train_batch_size", "temperature", "cl_sim_weight", "cl_org_weight",
    "cl_decoder_weight", "item_cl_weight",
)


@dataclass(frozen=True)
class Direction:
    pair: str
    name: str
    source: str
    target: str
    parameters: tuple
    provenance: str


def make_pair(pair, first, second, first_dataset, second_dataset,
              forward, reverse, provenance):
    return (
        Direction(pair, first + "_to_" + second, first_dataset, second_dataset,
                  forward, provenance),
        Direction(pair, second + "_to_" + first, second_dataset, first_dataset,
                  reverse, provenance),
    )


# Ogni tupla: batch, temperatura, cl_sim, cl_org, cl_decoder, item_cl.
GROUPS = {
    "original": (
        *make_pair(
            "douban_movie_book_10core", "movie", "book",
            "DoubanMovie_DoubanBook_commonUser_10-core",
            "DoubanBook_DoubanMovie_commonUser_10-core",
            (4096, 0.10, 0.01, 1, 0.01, 0.01),
            (2048, 0.30, 0.01, 1, 0.01, 0.01),
            "Autori: results & hypara setting/Douban-Movie&Book.md",
        ),
        *make_pair(
            "amazon_elec_cloth_10core", "elec", "cloth",
            "AmazonElec_AmazonCloth_commonUser_10-core",
            "AmazonCloth_AmazonElec_commonUser_10-core",
            (4096, 0.15, 0.10, 0.10, 0.10, 0.10),
            (4096, 0.15, 0.01, 1, 0.10, 0.01),
            "Autori: results & hypara setting/Amazon-Elec&Cloth.md",
        ),
        *make_pair(
            "amazon_sport_cloth_5core", "sport", "cloth",
            "AmazonSport_AmazonCloth_commonUser_5-core",
            "AmazonCloth_AmazonSport_commonUser_5-core",
            (2048, 0.30, 0.10, 0.10, 0.10, 0.10),
            (2048, 0.25, 0.01, 1, 0.10, 0.01),
            "Autori: results & hypara setting/Amazon-Sport&Cloth.md",
        ),
    ),
    "extension": (
        *make_pair(
            "amazon_beauty_health_5core", "beauty", "health",
            "Beauty_and_Personal_Care_Health_and_Household_commonUser_5-core",
            "Health_and_Household_Beauty_and_Personal_Care_commonUser_5-core",
            (4096, 0.25, 0.01, 1, 0.10, 0.01),
            (4096, 0.30, 0.10, 0.10, 0.10, 0.10),
            "Log VM storici Beauty-Health / Health-Beauty (tutti i cinque seed)",
        ),
        *make_pair(
            "amazon_garden_pet_5core", "garden", "pet",
            "Patio_Lawn_and_Garden_Pet_Supplies_commonUser_5-core",
            "Pet_Supplies_Patio_Lawn_and_Garden_commonUser_5-core",
            (4096, 0.25, 0.01, 1, 0.10, 0.01),
            (4096, 0.30, 0.10, 0.10, 0.10, 0.10),
            "Log VM storici Garden-Pet / Pet-Garden (tutti i cinque seed)",
        ),
        *make_pair(
            "amazon_beauty_pet_5core", "beauty", "pet",
            "Beauty_and_Personal_Care_Pet_Supplies_commonUser_5-core",
            "Pet_Supplies_Beauty_and_Personal_Care_commonUser_5-core",
            (4096, 0.30, 0.10, 0.10, 0.10, 0.10),
            (4096, 0.30, 0.10, 0.10, 0.10, 0.10),
            "Log VM storici Beauty-Pet / Pet-Beauty (tutti i cinque seed)",
        ),
    ),
}


def resolve_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def run_directory(output_root, direction, seed):
    return output_root / direction.pair / direction.name / "DGCDR" / str(seed)


def run_config(direction, seed, data_root, directory, use_gpu=True):
    return {
        **COMMON_HPARAMS,
        **dict(zip(PARAMETER_NAMES, direction.parameters)),
        "seed": seed,
        "source_domain": {"dataset": direction.source, "data_path": str(data_root)},
        "target_domain": {"dataset": direction.target, "data_path": str(data_root)},
        "data_path": str(data_root),
        "checkpoint_dir": str(directory),
        "use_gpu": use_gpu,
        "reproducibility": True,
        "state": "INFO",
        "show_progress": False,
        "log_wandb": False,
        "save_dataset": False,
        "dataset_save_path": None,
        "save_dataloaders": False,
        "dataloaders_save_path": None,
    }


def config_signature(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in CONFIG_FILES:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def is_completed(directory, signature):
    log_path = directory / "train.log"
    if not log_path.exists() or not log_path.stat().st_size:
        if any(directory.glob("*.pth")):
            raise RuntimeError("Checkpoint senza log del wrapper in " + str(directory))
        return False
    recorded, last_line = None, ""
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            last_line = line.strip()
            if line.startswith("RUN_CONFIG_SHA256 "):
                recorded = line.strip().split()[1]
    if recorded != signature:
        raise RuntimeError(
            "Configurazione diversa o log non riconosciuto in %s. "
            "Usa una nuova cartella --output-root." % directory
        )
    checkpoints = list(directory.glob("DGCDR-*.pth"))
    return (len(checkpoints) == 1 and checkpoints[0].stat().st_size > 0
            and last_line == "RUN_COMPLETED " + checkpoints[0].name)


class NoTensorboard:
    """Disabilita i file eventi: per questi run bastano log e pesi."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_hparams(self, *args, **kwargs):
        pass


def execute_training(direction, seed, data_root, directory, use_gpu=True):
    # RecBole interpreta anche sys.argv: gli argomenti del wrapper sono gia letti.
    sys.argv = [sys.argv[0]]
    quick_start = importlib.import_module("recbole_cdr.quick_start.quick_start")
    trainer_module = importlib.import_module("recbole.trainer.trainer")
    logger_module = importlib.import_module("recbole.utils.logger")

    def init_run_logger(config):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler.addFilter(logger_module.RemoveColorFilter())
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    # Modifiche solo in memoria, nel processo del singolo run.
    quick_start.init_logger = init_run_logger
    trainer_module.get_tensorboard = lambda logger: NoTensorboard()
    print("Provenienza iperparametri: " + direction.provenance, flush=True)
    print("Protocollo codice esistente: BOTH; source non suddiviso.", flush=True)
    quick_start.run_recbole_cdr(
        model="DGCDR",
        config_file_list=[str(path) for path in CONFIG_FILES],
        config_dict=run_config(direction, seed, data_root, directory, use_gpu),
        saved=True,
    )


def run_one(direction, seed, data_root, output_root, use_gpu=True):
    directory = run_directory(output_root, direction, seed)
    signature = config_signature(run_config(direction, seed, data_root, directory, use_gpu))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "train.log").open("a+", encoding="utf-8", buffering=1) as log:
        try:
            fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Run gia in esecuzione: " + str(directory))
        if is_completed(directory, signature):
            print("SALTA completato: %s seed=%s" % (direction.name, seed), flush=True)
            return
        # Solo output incompleti di questo wrapper, riconosciuti dalla firma nel log.
        for checkpoint in directory.glob("DGCDR-*.pth"):
            checkpoint.unlink()
        log.write("RUN_CONFIG_SHA256 " + signature + "\n")
        log.write("RUN_STARTED_FROM_ZERO " + datetime.now().isoformat() + "\n")
        log.flush()
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker", direction.name, str(seed),
            "--data-root", str(data_root), "--output-root", str(output_root),
        ]
        if not use_gpu:
            command.append("--cpu")
        print("AVVIA: %s seed=%s -> %s" % (direction.name, seed, directory), flush=True)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=environment, pass_fds=(log.fileno(),),
        )
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.write("RUN_INTERRUPTED\n")
            raise
        finally:
            process.stdout.close()
        checkpoints = list(directory.glob("DGCDR-*.pth"))
        if returncode or len(checkpoints) != 1 or checkpoints[0].stat().st_size == 0:
            log.write("RUN_FAILED exit_code=%s\n" % returncode)
            raise RuntimeError("Run non completato. Consulta %s e rilancia lo stesso comando."
                               % (directory / "train.log"))
        log.write("RUN_COMPLETED " + checkpoints[0].name + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Mostra il piano senza scrivere o allenare.")
    parser.add_argument("--cpu", action="store_true", help="Usa la CPU invece della GPU.")
    parser.add_argument("--worker", nargs=2, metavar=("DIRECTION", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    data_root, output_root = resolve_path(args.data_root), resolve_path(args.output_root)
    if args.worker:
        directions = {d.name: d for group in GROUPS.values() for d in group}
        direction, seed = directions[args.worker[0]], int(args.worker[1])
        execute_training(direction, seed, data_root,
                         run_directory(output_root, direction, seed), not args.cpu)
        return 0
    if GROUP not in GROUPS:
        parser.error('GROUP deve essere "original" oppure "extension".')
    if not SEEDS or len(SEEDS) != len(set(SEEDS)):
        parser.error("SEEDS deve contenere seed distinti.")

    print("Gruppo %s: %d direzioni x %d seed = %d run." %
          (GROUP, len(GROUPS[GROUP]), len(SEEDS), len(GROUPS[GROUP]) * len(SEEDS)))
    pending, missing = [], set()
    for direction in GROUPS[GROUP]:
        if args.dry_run:
            print("\n%s: %s -> %s" % (direction.name, direction.source, direction.target))
            print("  " + str(dict(zip(PARAMETER_NAMES, direction.parameters))))
        for seed in SEEDS:
            directory = run_directory(output_root, direction, seed)
            signature = config_signature(run_config(direction, seed, data_root, directory, not args.cpu))
            complete = is_completed(directory, signature)
            if args.dry_run or complete:
                print("  %s seed=%s: %s" % (direction.name, seed,
                      "SALTA completato" if complete else "DA ESEGUIRE"))
            if not complete:
                pending.append((direction, seed))
                for name in (direction.source, direction.target):
                    path = data_root / name / (name + ".inter")
                    if not path.is_file():
                        missing.add(path)
    if missing:
        message = "Dataset mancanti (controlla --data-root):\n" + "\n".join(map(str, sorted(missing)))
        if args.dry_run:
            print("\n" + message)
        else:
            raise FileNotFoundError(message)
    if args.dry_run:
        return 0
    for direction, seed in pending:
        run_one(direction, seed, data_root, output_root, not args.cpu)
    print("Gruppo %s completato: %d nuovi run, %d gia completati." %
          (GROUP, len(pending), len(GROUPS[GROUP]) * len(SEEDS) - len(pending)))
    return 0


if __name__ == "__main__":
    worker_process = "--worker" in sys.argv
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrotto. Rilancia lo stesso comando per riprendere la serie.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        if worker_process:
            raise
        print("ERRORE: " + str(error), file=sys.stderr)
        sys.exit(1)
