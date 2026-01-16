from visualization.charts import model_comparison_chart, drift_heatmap

def prepare_dashboard(payload):
    return {
        "model_chart": model_comparison_chart(payload["model_metrics"]),
        "drift_chart": drift_heatmap(payload["drift"])
    }
