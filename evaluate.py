from evaluation import run_evaluation


if __name__ == "__main__":
    report = run_evaluation()
    print(f"Evaluation complete: {report['sample_count']} samples")
    print(f"Accuracy: {report['accuracy']}%")
    print(f"Model version: {report.get('model_version')}")
    print(f"Dataset version: {report.get('dataset_version')}")
    print(f"Resolved accuracy: {report['resolved_accuracy']}%")
    print(f"Coverage: {report['coverage']}%")
    print(f"ROC-AUC: {report['roc_auc']}%")
    print(f"False positive rate: {report['false_positive_rate']}%")
    print(f"False negative rate: {report['false_negative_rate']}%")
    print(f"Hard examples exported to: {report.get('hard_example_export_path')}")
