import os
import json
import csv
from datetime import datetime


class ExperimentLogger:
    def __init__(self, base_dir="logs", exp_id=None, config_dict=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = exp_id if exp_id is not None else f"EXP_{timestamp}"
        self.run_dir = os.path.join(base_dir, self.exp_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.train_log_path = os.path.join(self.run_dir, "train_log.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.result_csv_path = os.path.join(self.run_dir, "result.csv")
        self.note_path = os.path.join(self.run_dir, "notes.txt")

        self.summary = {
            "exp_id": self.exp_id,
            "created_at": timestamp,
            "config": config_dict if config_dict is not None else {},
            "per_domain_acc": {},
            "avg_acc": None,
            "best_epochs": {},
            "status": "running",
        }

        if config_dict is not None:
            with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def log_train(self, record: dict):
        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_domain_result(self, domain_name, acc):
        self.summary["per_domain_acc"][domain_name] = float(acc)

    def log_best_epoch(self, domain_name, epoch):
        self.summary["best_epochs"][domain_name] = int(epoch)

    def set_avg_acc(self, avg_acc):
        self.summary["avg_acc"] = float(avg_acc)

    def set_status(self, status):
        self.summary["status"] = status

    def add_note(self, text):
        with open(self.note_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def save_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

    def append_result_csv(self):
        file_exists = os.path.exists(self.result_csv_path)

        fieldnames = ["exp_id", "status", "avg_acc"]
        domain_keys = sorted(self.summary["per_domain_acc"].keys())
        fieldnames += domain_keys

        row = {
            "exp_id": self.summary["exp_id"],
            "status": self.summary["status"],
            "avg_acc": self.summary["avg_acc"],
        }
        for k in domain_keys:
            row[k] = self.summary["per_domain_acc"][k]

        with open(self.result_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)