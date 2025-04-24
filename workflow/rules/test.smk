rule evaluate_model:
    input:
        model="results/model.pkl",
        test_data="results/test_data.pkl"
    output:
        "results/metrics.txt"
    script:
        "../scripts/test.py"