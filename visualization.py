import os
import sys


def show_compared_test_table_prettytable(result_objects):
    import prettytable

    table = prettytable.PrettyTable()
    table.field_names = [
        "Method",
        "Accuracy",
        "Macro F1",
        "Macro Precision",
        "Macro Recall",
    ]

    for method_name, method_results in result_objects.items():
        test_results = method_results["test_results"]
        table.add_row(
            [
                method_name,
                f'{test_results["accuracy"]:.4f}',
                f'{test_results["macro_f1"]:.4f}',
                f'{test_results["macro_precision"]:.4f}',
                f'{test_results["macro_recall"]:.4f}',
            ]
        )
    print(table)
    # write the table to a text file
    with open("test_results.txt", "w") as file:
        file.write(str(table))


def draw_training_history(result_objects):
    import matplotlib.pyplot as plt

    num_methods = len(result_objects)
    fig, axes = plt.subplots(num_methods, 1, figsize=(15, 5 * num_methods))

    for i, (method_name, method_results) in enumerate(result_objects.items()):
        history = method_results["training_results"]["history"]
        axes[i].plot(history["epoch"], history["accuracy"], label="accuracy")
        axes[i].plot(history["epoch"], history["macro_f1"], label="macro_f1")
        axes[i].plot(
            history["epoch"], history["macro_precision"], label="macro_precision"
        )
        axes[i].plot(history["epoch"], history["macro_recall"], label="macro_recall")
        # axes[i].set_title(method_name)

        # Title is training history of model
        axes[i].set_title(f"Training history of {method_name}")

        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Validation Result of Validation Set")
        axes[i].legend()

    plt.tight_layout()
    # plt.show()

    # save the figure
    plt.savefig("training_history.png")
