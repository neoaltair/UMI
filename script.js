const API_BASE = "http://localhost:8000";

// State
let weightHistory = [];
let featCols = [];
let roundHistory = [];
let architectures = {};

// UI Elements
const logContainer = document.getElementById("log-container");
const startBtn = document.getElementById("start-train-btn");
const archContainer = document.getElementById("arch-container");

// View Switching
document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", () => {
        document.querySelector(".nav-item.active").classList.remove("active");
        item.classList.add("active");

        const role = item.getAttribute("data-role");
        updateView(role);
    });
});

function updateView(role) {
    const researcherView = document.querySelectorAll(".dashboard-grid")[0];
    const doctorView = document.getElementById("doctor-view");
    const title = document.getElementById("view-title");
    const subtitle = document.getElementById("view-subtitle");

    if (role === "Doctor") {
        researcherView.style.display = "none";
        doctorView.style.display = "grid";
        title.innerText = "Patient Triage Center";
        subtitle.innerText = "Global FedProx model scores risk → Unified Clinical Pathway";
        initDoctorForm();
    } else {
        researcherView.style.display = "grid";
        doctorView.style.display = "none";
        title.innerText = role === "Researcher" ? "Federated Research Intelligence" : "Privacy & Governance Command Center";
        subtitle.innerText = "Cross-silo knowledge synthesis with differential privacy";
    }
}

// Training Logic
startBtn.addEventListener("click", () => {
    const n_rounds = document.getElementById("n_rounds").value;
    const mu = document.getElementById("mu").value;
    const epsilon = document.getElementById("epsilon").value;

    logContainer.innerHTML = "";
    addLog("Initiating Federated Training Session...", "system");

    document.getElementById("status-badge").innerHTML = `Status: <span style="color: var(--accent-purple-bright); animation: pulse 2s infinite;">● Syncing</span>`;
    startBtn.disabled = true;

    const eventSource = new EventSource(`${API_BASE}/stream-train?n_rounds=${n_rounds}&mu=${mu}&epsilon=${epsilon}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "local_training") {
            addLog(data.event, "local_training", data.hospital);
        } else if (data.type === "anomaly") {
            addLog(data.event, "anomaly", data.hospital);
        } else if (data.type === "complete") {
            addLog("Training sequence complete. Synchronizing results...", "complete");
            eventSource.close();
            fetchHistory();
            startBtn.disabled = false;
            document.getElementById("status-badge").innerHTML = `Status: <span style="color: var(--accent-green-bright);">● Synced</span>`;
        } else if (data.type === "error") {
            addLog(`Error: ${data.message}`, "anomaly");
            eventSource.close();
            startBtn.disabled = false;
        }
    };
});

function addLog(message, type, hospital = "") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    let tag = type === "local_training" ? "TX" : type === "anomaly" ? "ALERT" : "SYS";
    if (hospital) tag = hospital.substring(0, 3).toUpperCase();

    entry.innerHTML = `<span class="log-time">[${time}]</span><span class="log-tag">[${tag}]</span> ${message}`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

async function fetchHistory() {
    const resp = await fetch(`${API_BASE}/history`);
    const data = await resp.json();

    weightHistory = data.weight_history;
    featCols = data.feat_cols;
    roundHistory = data.round_history;
    architectures = data.architectures;

    updateKPIs();
    renderArchitecture();
    animateTrajectory();
}

function updateKPIs() {
    const lastRound = roundHistory[roundHistory.length - 1];
    document.getElementById("kpi-hospitals").innerText = Object.keys(architectures).length;
    document.getElementById("kpi-accuracy").innerText = `${lastRound.Accuracy}%`;
    document.getElementById("kpi-epsilon").innerText = lastRound.Epsilon_Spent.toFixed(2);
}

function renderArchitecture() {
    archContainer.innerHTML = "";
    for (const [hosp, models] of Object.entries(architectures)) {
        const card = document.createElement("div");
        card.className = "hospital-arch-card";

        let modelItems = "";
        for (const [mName, acc] of Object.entries(models)) {
            modelItems += `
                <div class="model-item">
                    <span>${mName}</span>
                    <span class="acc-val">${(acc * 100).toFixed(1)}%</span>
                </div>`;
        }

        card.innerHTML = `
            <div class="hospital-name">
                ${hosp}
                <span style="font-size: 0.6rem; color: var(--accent-blue-bright);">VERIFIED</span>
            </div>
            <div class="model-list">
                ${modelItems}
            </div>
        `;
        archContainer.appendChild(card);
    }
}

function animateTrajectory() {
    // We'll use Plotly to animate the top feature's coefficient over rounds
    const topFeatureIdx = 0; // Just picking first for demonstration
    const featName = featCols[topFeatureIdx];

    const hospitals = Object.keys(weightHistory[0].hospitals);
    const traces = hospitals.map(h => ({
        x: weightHistory.map(r => r.round),
        y: weightHistory.map(r => r.hospitals[h][topFeatureIdx]),
        name: h,
        mode: 'lines+markers',
        type: 'scatter',
        line: { width: 3 }
    }));

    traces.push({
        x: weightHistory.map(r => r.round),
        y: weightHistory.map(r => r.global[topFeatureIdx]),
        name: 'GLOBAL CONVERGENCE',
        mode: 'lines+markers',
        type: 'scatter',
        line: { width: 5, color: '#white', dash: 'dash' }
    });

    const layout = {
        title: `Weight Trajectory: ${featName} Evolution`,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#c9d1d9' },
        xaxis: { title: 'Communication Round', gridcolor: '#21262d' },
        yaxis: { title: 'Coefficient Value', gridcolor: '#21262d' },
        margin: { t: 40, b: 40, l: 40, r: 40 },
        legend: { x: 0, y: 1 }
    };

    Plotly.newPlot('trajectory-chart', traces, layout);
}

// Doctor View Logic
function initDoctorForm() {
    const form = document.getElementById("vitals-form");
    if (form.innerHTML !== "") return;

    const fields = [
        { id: "age", label: "Age", val: 54 },
        { id: "trestbps", label: "Resting BP", val: 130 },
        { id: "chol", label: "Cholesterol", val: 246 },
        { id: "thalch", label: "Max Heart Rate", val: 150 },
        { id: "oldpeak", label: "ST Depr", val: 1.0 },
        { id: "ca", label: "Vessels (0-3)", val: 0 },
        { id: "sex", label: "Sex (0=F, 1=M)", val: 1 },
        { id: "cp", label: "Chest Pain (1-4)", val: 1 },
        { id: "fbs", label: "Sugar > 120 (0/1)", val: 0 },
        { id: "restecg", label: "ECG (0-2)", val: 0 },
        { id: "exang", label: "Ex Angina (0/1)", val: 1 }
    ];

    fields.forEach(f => {
        form.innerHTML += `
            <div class="form-group">
                <label>${f.label}</label>
                <input type="number" id="v_${f.id}" value="${f.val}">
            </div>
        `;
    });
}

document.getElementById("predict-btn").addEventListener("click", async () => {
    const vitals = {};
    const fieldIds = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "sex", "cp", "fbs", "restecg", "exang"];
    fieldIds.forEach(id => {
        vitals[id] = parseFloat(document.getElementById(`v_${id}`).value);
    });

    const resp = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vitals })
    });
    const data = await resp.json();

    renderRisk(data.probability);
    renderContribution(data.contributions);
});

function renderRisk(prob) {
    const container = document.getElementById("risk-gauge-container");
    const color = prob > 60 ? "#f85149" : prob > 30 ? "#d29922" : "#3fb950";

    // Simple custom gauge using Plotly
    const trace = {
        type: "indicator",
        mode: "gauge+number",
        value: prob,
        title: { text: "Cardiac Risk Score", font: { size: 18, color: '#c9d1d9' } },
        gauge: {
            axis: { range: [0, 100], tickwidth: 1, tickcolor: "#c9d1d9" },
            bar: { color: color },
            bgcolor: "#0d1117",
            borderwidth: 2,
            bordercolor: "#21262d",
            steps: [
                { range: [0, 30], color: "rgba(63, 185, 80, 0.1)" },
                { range: [30, 60], color: "rgba(210, 153, 34, 0.1)" },
                { range: [60, 100], color: "rgba(248, 81, 73, 0.1)" }
            ]
        }
    };

    const layout = {
        width: 400, height: 250, margin: { t: 40, b: 0, l: 50, r: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#c9d1d9' }
    };

    Plotly.newPlot(container, [trace], layout);
}

function renderContribution(contributions) {
    const container = document.getElementById("contribution-report");
    let html = `
        <h4 style="font-size: 0.8rem; margin-bottom: 1rem; color: var(--text-muted);">HOSPITAL CONTRIBUTION BREAKDOWN</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
    `;

    for (const [hosp, info] of Object.entries(contributions)) {
        const pct = (info.weight * 100).toFixed(1);
        html += `
            <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 8px;">
                <div style="width: 40px; font-weight: 800; color: var(--accent-blue-bright);">${pct}%</div>
                <div style="flex: 1; font-size: 0.85rem;">${hosp}</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">n=${info.sample_size}</div>
            </div>
        `;
    }

    html += `</div>`;
    container.innerHTML = html;
}
