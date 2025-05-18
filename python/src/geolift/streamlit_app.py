import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from geolift.data import GeoData
from geolift.market_selection import market_correlations, market_selection
from geolift.multicell import multicell_market_selection
from geolift.modeling import BayesianSCM

st.title("GeoLift Multi-Cell Demo")

uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Data preview", df.head())

    k = st.number_input("Number of cells", min_value=1, value=2, step=1)
    top_n = st.number_input("Top N correlated markets", min_value=1, value=5, step=1)

    geo = GeoData.read(df)

    # Market correlations heatmap
    corr = market_correlations(geo.data)
    pivot = corr.pivot("var1", "var2", "correlation")
    fig, ax = plt.subplots()
    im = ax.imshow(pivot, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("Market correlations")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    mc = multicell_market_selection(geo, k=k, top_n=top_n)
    st.subheader("Market cells")
    for cell_id, cell in mc.cells.items():
        st.write(f"Cell {cell_id}: {', '.join(cell.locations)}")
        st.dataframe(cell.selection)

    # Build simple synthetic control for first cell as example
    first_cell = mc.cells[1]
    treatment_locs = first_cell.locations[:1]
    control_locs = [loc for loc in first_cell.locations if loc not in treatment_locs]
    df_sub = geo.data[geo.data['location'].isin(first_cell.locations)]
    df_sub['is_treatment'] = df_sub['location'].isin(treatment_locs)

    control = df_sub[~df_sub['is_treatment']].pivot(index='time', columns='location', values='Y')
    treatment = df_sub[df_sub['is_treatment']].groupby('time')['Y'].sum().reset_index()

    model = BayesianSCM(control, treatment)
    trace = model.fit()

    beta_samples = trace.posterior['beta']
    mean_beta = beta_samples.mean(dim=("chain", "draw")).values
    st.subheader("Synthetic control coefficients")
    st.write(pd.Series(mean_beta, index=control.columns))

    pred = model.predict(control)
    # credible interval
    beta_reshaped = beta_samples.stack(sample=("chain","draw"))
    preds = control.values @ beta_reshaped.values
    lower = pd.Series(preds.mean(axis=1) - preds.std(axis=1)*1.96)
    upper = pd.Series(preds.mean(axis=1) + preds.std(axis=1)*1.96)

    times = treatment['time']
    fig2, ax2 = plt.subplots()
    ax2.plot(times, treatment['Y'], label='Observed')
    ax2.plot(times, pred, label='Synthetic')
    ax2.fill_between(times, lower, upper, color='gray', alpha=0.3, label='95% CI')
    ax2.legend()
    ax2.set_title('Synthetic Control with Uncertainty')
    st.pyplot(fig2)

