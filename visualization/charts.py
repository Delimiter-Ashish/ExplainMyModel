import plotly.express as px
import plotly.figure_factory as ff

def model_comparison_chart(metrics):
    df = []
    for m, v in metrics.items():
        df.append({"model": m, "accuracy": v["accuracy"], "f1": v["f1"]})

    fig = px.bar(df, x="model", y=["accuracy","f1"], barmode="group")
    return fig.to_html(full_html=False)

def drift_heatmap(drift):
    import pandas as pd
    df = pd.DataFrame.from_dict(drift, orient="index")
    fig = ff.create_annotated_heatmap(
        z=df[["ks_stat"]].values,
        x=["KS Stat"],
        y=list(df.index)
    )
    return fig.to_html(full_html=False)
