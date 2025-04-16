import streamlit as st
import pandas as pd
import numpy as np
import wntr
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from functools import partial
from io import BytesIO
import tempfile
import shutil
import random

st.set_page_config(page_title="EPANET Calibration Tool", layout="wide")
st.title("EPANET Calibration + INP Update Tool (Web Edition)")

# File Upload
inp_file = st.file_uploader("Step 1: Upload EPANET .inp file", type=["inp"])
obs_file = st.file_uploader("Upload Observed Data (CSV)", type=["csv"])

st.session_state.setdefault("calibrated_df", None)
st.session_state.setdefault("rmse_results", {})
st.session_state.setdefault("last_wn", None)

# Global objective function
def objective_wrapper(params, wn_path, link_names, obs_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".inp") as tf:
        shutil.copyfile(wn_path, tf.name)
        temp_inp = tf.name

    try:
        wn = wntr.network.WaterNetworkModel(temp_inp)
        for i, link_name in enumerate(link_names):
            wn.get_link(link_name).roughness = params[i]

        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        sim_pressures = results.node.get('pressure', pd.DataFrame())
        sim_flows = results.link.get('flowrate', pd.DataFrame())

        error = 0.0
        for col in obs_data.columns:
            if col in sim_pressures.columns:
                sim_series = sim_pressures[col]
            elif col in sim_flows.columns:
                sim_series = sim_flows[col] * 1000
            else:
                continue
            aligned = sim_series.reindex(obs_data.index).interpolate()
            relative_error = ((obs_data[col] - aligned) / obs_data[col]).dropna()
            error += np.nansum(relative_error**2)

        return error

    except Exception:
        return np.inf

# Calibration
def run_calibration(wn, wn_path, obs_data):
    observed_nodes = set(obs_data.columns)

    all_valid_links = [
        l for l in wn.link_name_list
        if isinstance(wn.get_link(l), wntr.network.elements.Pipe) and (
            wn.get_link(l).start_node_name in observed_nodes or
            wn.get_link(l).end_node_name in observed_nodes
        )
    ]

    subset_links = all_valid_links

    st.write(f"🔧 Calibrating {len(subset_links)} pipes connected to observed nodes.")

    bounds = [(65, 150)] * len(subset_links)
    objective = partial(objective_wrapper, wn_path=wn_path, link_names=subset_links, obs_data=obs_data)

    result = differential_evolution(objective, bounds, maxiter=10, workers=1)
    best_params = result.x

    local_result = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds)
    refined_params = local_result.x

    for i, link_name in enumerate(subset_links):
        wn.get_link(link_name).roughness = refined_params[i]

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    selected_df = pd.DataFrame({
        "Pipe": subset_links,
        "Calibrated Roughness": refined_params
    })
    st.session_state["selected_pipes_df"] = selected_df

    return wn, results

# RMSE

def compute_rmse(obs_data, results):
    sim_pressures = results.node.get('pressure', pd.DataFrame())
    sim_flows = results.link.get('flowrate', pd.DataFrame()) * 1000
    rmse_results = {}
    messages = []

    for col in obs_data.columns:
        if col in sim_pressures.columns:
            sim_series = sim_pressures[col]
            unit = "m"
        elif col in sim_flows.columns:
            sim_series = sim_flows[col]
            unit = "L/s"
        else:
            continue
        aligned = sim_series.reindex(obs_data.index).interpolate()
        diff = obs_data[col] - aligned
        rmse = np.sqrt(np.nanmean(diff**2)) if not diff.dropna().empty else np.nan
        rmse_results[col] = rmse
        messages.append(f"RMSE {col}: {rmse:.2f} {unit}")

    return rmse_results, messages

# Plotting

def plot_results(obs_data, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for col in obs_data.columns:
        if col in results.node.get('pressure', pd.DataFrame()).columns:
            sim_series = results.node['pressure'][col]
            aligned = sim_series.reindex(obs_data.index).interpolate()
            ax1.plot(obs_data.index, obs_data[col], label=f"Obs {col}")
            ax1.plot(obs_data.index, aligned, label=f"Sim {col}")
        elif col in results.link.get('flowrate', pd.DataFrame()).columns:
            sim_series = results.link['flowrate'][col] * 1000
            aligned = sim_series.reindex(obs_data.index).interpolate()
            ax2.plot(obs_data.index, obs_data[col], label=f"Obs {col}")
            ax2.plot(obs_data.index, aligned, label=f"Sim {col}")

    ax1.set_title("Observed vs Simulated Pressure")
    ax1.set_ylabel("Pressure (m)")
    ax1.legend()
    ax2.set_title("Observed vs Simulated Flow")
    ax2.set_ylabel("Flow (L/s)")
    ax2.set_xlabel("Time")
    ax2.legend()

    st.pyplot(fig)

# Network map of calibrated pipes
def plot_selected_network(wn, selected_df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    wntr.graphics.plot_network(wn, ax=ax)

    for _, row in selected_df.iterrows():
        link = wn.get_link(row["Pipe"])
        x = (link.start_node.coordinates[0] + link.end_node.coordinates[0]) / 2
        y = (link.start_node.coordinates[1] + link.end_node.coordinates[1]) / 2
        ax.text(x, y, f"{row['Calibrated Roughness']:.1f}", color="red", fontsize=8, ha="center")

    plt.tight_layout()
    st.pyplot(fig)

# Main execution
if inp_file and obs_file:
    temp_path = "temp_model.inp"
    with open(temp_path, "wb") as f:
        f.write(inp_file.getvalue())

    obs_data = pd.read_csv(obs_file, index_col=0)
    wn = wntr.network.WaterNetworkModel(temp_path)

    if st.button("Run Calibration"):
        with st.spinner("Running calibration. Please wait..."):
            wn, results = run_calibration(wn, temp_path, obs_data)
            rmse, messages = compute_rmse(obs_data, results)
            st.session_state["rmse_results"] = rmse
            st.session_state["last_wn"] = wn

            for msg in messages:
                st.write(msg)

            plot_results(obs_data, results)

            calibrated_data = [
                [link_name, link.roughness]
                for link_name in wn.link_name_list
                if isinstance((link := wn.get_link(link_name)), wntr.network.elements.Pipe)
            ]
            st.session_state["calibrated_df"] = pd.DataFrame(calibrated_data, columns=["Pipe", "Calibrated Roughness"])

            if "selected_pipes_df" in st.session_state:
                st.subheader("📋 Pipes Selected for Calibration")
                st.dataframe(st.session_state["selected_pipes_df"])

                st.subheader("🌐 Network Map with Calibrated Pipes")
                plot_selected_network(wn, st.session_state["selected_pipes_df"])

# Download CSV
if st.session_state["calibrated_df"] is not None:
    st.download_button("Download Calibrated Roughness CSV", st.session_state["calibrated_df"].to_csv(index=False), file_name="calibrated_roughness.csv")

# Update INP file
st.subheader("Step 2: Update INP File with Calibrated Roughness")
roughness_csv = st.file_uploader("Upload Roughness CSV (Pipe, Calibrated Roughness)", type=["csv"], key="update_csv")
inp_to_update = st.file_uploader("Upload INP File to Update", type=["inp"], key="update_inp")

if roughness_csv and inp_to_update:
    rough_df = pd.read_csv(roughness_csv)
    rough_dict = dict(zip(rough_df['Pipe'], rough_df['Calibrated Roughness']))

    inp_lines = inp_to_update.getvalue().decode("utf-8").splitlines()
    start, end = None, None
    for i, line in enumerate(inp_lines):
        if line.strip().upper() == "[PIPES]":
            start = i
        elif start and line.strip().startswith("[") and line.strip().endswith("]"):
            end = i
            break

    for i in range(start + 2, end):
        parts = inp_lines[i].split()
        if not parts or parts[0].startswith(";"):
            continue
        pipe_id = parts[0]
        if pipe_id in rough_dict:
            parts[5] = f"{rough_dict[pipe_id]:.6f}"
            inp_lines[i] = "\t".join(parts)

    updated_content = "\n".join(inp_lines)
    st.download_button(
        label="Download Updated INP File",
        data=BytesIO(updated_content.encode("utf-8")),
        file_name="updated_model.inp",
        mime="text/plain"
    )
