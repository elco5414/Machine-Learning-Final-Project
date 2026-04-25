const API_BASE = "";

// --- header date ---
const now = new Date();
document.getElementById("date-line").textContent = now.toLocaleDateString(
  "en-US",
  { weekday: "short", month: "short", day: "numeric", year: "numeric" },
);

// ===================================================
//   PORTFOLIO ANALYZER
// ===================================================
const holdings = {};
const $holdings = document.getElementById("holdings");
const $analyze = document.getElementById("analyze-btn");
const $clear = document.getElementById("clear-btn");
const $portError = document.getElementById("port-error");
const $portResults = document.getElementById("port-results");
const $portHorizon = document.getElementById("port-horizon");

function renderHoldings() {
  const entries = Object.entries(holdings);
  if (entries.length === 0) {
    $holdings.innerHTML = `<div class="empty">No holdings yet — add a ticker to begin.</div>`;
    $analyze.disabled = true;
    $clear.disabled = true;
    return;
  }
  $holdings.innerHTML = entries
    .map(([t, s]) => `
      <div class="holding-row">
        <div class="ticker">${t}</div>
        <div class="shares">${s} ${s === 1 ? "share" : "shares"}</div>
        <button class="remove-btn" data-ticker="${t}" title="Remove">×</button>
      </div>
    `).join("");
  $analyze.disabled = false;
  $clear.disabled = false;
  $holdings.querySelectorAll(".remove-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      delete holdings[btn.dataset.ticker];
      renderHoldings();
    });
  });
}

document.getElementById("add-holding").addEventListener("click", () => {
  const ticker = document
    .getElementById("port-ticker")
    .value.trim()
    .toUpperCase();
  const shares = parseInt(document.getElementById("port-shares").value, 10);
  if (!ticker || !shares || shares < 1) {
    showPortError("Enter a ticker and a share count.");
    return;
  }
  if (!tickersLoaded) {
    showPortError("Still loading ticker list — try again in a moment.");
    return;
  }
  if (!validTickers.has(ticker)) {
    showPortError(
      `"${ticker}" is not in the S&P 500. Use the ticker symbol (e.g. AAPL, MSFT, BRK-B).`,
    );
    return;
  }
  holdings[ticker] = (holdings[ticker] || 0) + shares;
  document.getElementById("port-ticker").value = "";
  document.getElementById("port-shares").value = "";
  document.getElementById("port-ticker").focus();
  clearPortError();
  renderHoldings();
});

$clear.addEventListener("click", () => {
  Object.keys(holdings).forEach((k) => delete holdings[k]);
  $portResults.style.display = "none";
  clearPortError();
  renderHoldings();
});

$analyze.addEventListener("click", async () => {
  clearPortError();
  const horizon = $portHorizon.value;
  $analyze.innerHTML = `<span class="loading"></span>Analyzing…`;
  $analyze.disabled = true;
  try {
    const res = await fetch(API_BASE + "/portfolio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolio: holdings, horizon }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    renderPortfolioResults(await res.json());
  } catch (e) {
    showPortError(e.message || "Something went wrong.");
  } finally {
    $analyze.innerHTML = "Analyze portfolio";
    $analyze.disabled = false;
  }
});

function renderPortfolioResults(data) {
  const { portfolio, summary } = data;
  const summaryHtml = `
    <div class="summary-row">
      <div class="summary-cell">
        <div class="k">Total value</div>
        <div class="v">$${fmt(summary.total_value)}</div>
      </div>
      <div class="summary-cell">
        <div class="k">Weighted return (${summary.horizon_days}d)</div>
        <div class="v ${signClass(summary.weighted_return_pct)}">
          ${fmtPct(summary.weighted_return_pct)}
        </div>
      </div>
      <div class="summary-cell">
        <div class="k">Projected change</div>
        <div class="v mono ${signClass(summary.projected_dollar_change)}">
          ${fmtSignedDollars(summary.projected_dollar_change)}
        </div>
      </div>
    </div>
  `;
  document.getElementById("summary-card").innerHTML = summaryHtml;

  const rows = Object.entries(portfolio).map(([ticker, p]) => {
    if (p.predicted_return_pct === null || p.predicted_return_pct === undefined) {
      return `
        <div class="result-row">
          <div class="ticker">${ticker}</div>
          <div class="num" style="color:var(--ink-fade);">${p.shares} sh</div>
          <div class="num" style="color:var(--ink-fade);">—</div>
          <div class="pct" style="color:var(--ink-fade);">—</div>
          <div class="rec" style="color:var(--ink-fade);">no data</div>
        </div>`;
    }
    return `
      <div class="result-row">
        <div class="ticker">${ticker}</div>
        <div class="num">${p.shares} sh</div>
        <div class="num">$${fmt(p.position_value)}</div>
        <div class="pct ${signClass(p.predicted_return_pct)}">
          ${fmtPct(p.predicted_return_pct)}
        </div>
        <div class="rec ${signClass(p.predicted_return_pct)}">
          ${p.recommendation}
        </div>
      </div>`;
  }).join("");
  document.getElementById("result-rows").innerHTML = rows;
  $portResults.style.display = "block";
}

function showPortError(msg) {
  $portError.innerHTML = `<div class="error">${msg}</div>`;
}
function clearPortError() {
  $portError.innerHTML = "";
}

// ===================================================
//   PRICE FORECASTER
// ===================================================
const $fcTicker = document.getElementById("fc-ticker");
const $fcDate = document.getElementById("fc-date");
const $fcBtn = document.getElementById("forecast-btn");
const $fcError = document.getElementById("fc-error");
const $fcCard = document.getElementById("forecast-card");

// date input bounds
const tomorrow = new Date();
tomorrow.setDate(tomorrow.getDate() + 1);
const maxDate = new Date();
maxDate.setDate(maxDate.getDate() + 126);
$fcDate.min = tomorrow.toISOString().slice(0, 10);
$fcDate.max = maxDate.toISOString().slice(0, 10);
const defaultDate = new Date();
defaultDate.setDate(defaultDate.getDate() + 30);
$fcDate.value = defaultDate.toISOString().slice(0, 10);

// Shared set of valid S&P 500 tickers
const validTickers = new Set();
let tickersLoaded = false;

// load tickers
(async () => {
  try {
    const res = await fetch(API_BASE + "/tickers");
    const { tickers } = await res.json();
    tickers.forEach((t) => validTickers.add(t));
    tickersLoaded = true;
    $fcTicker.innerHTML =
      `<option value="">Select a ticker…</option>` +
      tickers.map((t) => `<option value="${t}">${t}</option>`).join("");
  } catch {
    $fcTicker.innerHTML = `<option value="">Could not load tickers</option>`;
  }
})();

$fcBtn.addEventListener("click", async () => {
  clearFcError();
  const ticker = $fcTicker.value;
  const date = $fcDate.value;
  if (!ticker) return showFcError("Pick a ticker.");
  if (!date) return showFcError("Pick a date.");

  $fcBtn.innerHTML = `<span class="loading"></span>Forecasting…`;
  $fcBtn.disabled = true;
  try {
    const res = await fetch(API_BASE + "/predict-price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, target_date: date }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    renderForecast(await res.json());
  } catch (e) {
    showFcError(e.message || "Something went wrong.");
    $fcCard.classList.remove("visible");
  } finally {
    $fcBtn.innerHTML = "Generate forecast";
    $fcBtn.disabled = false;
  }
});

function renderForecast(d) {
  document.getElementById("fc-current").innerHTML = splitPrice(d.current_price);
  document.getElementById("fc-projected").innerHTML = splitPrice(d.predicted_price);
  const chgEl = document.getElementById("fc-change");
  chgEl.textContent = fmtPct(d.predicted_return_pct);
  chgEl.className = "v " + signClass(d.predicted_return_pct);
  document.getElementById("fc-horizon").textContent =
    `${d.days_ahead} trading day${d.days_ahead === 1 ? "" : "s"}`;
  document.getElementById("fc-asof").textContent = d.as_of;
  $fcCard.classList.add("visible");
}

function showFcError(msg) {
  $fcError.innerHTML = `<div class="error">${msg}</div>`;
}
function clearFcError() {
  $fcError.innerHTML = "";
}

// ===================================================
//   MARKET SUGGESTIONS
// ===================================================
const $suggestionsBtn    = document.getElementById("suggestions-btn");
const $suggestionsGrid   = document.getElementById("suggestions-grid");
const $suggestionsError  = document.getElementById("suggestions-error");
const $suggestionsHorizon = document.getElementById("suggestions-horizon");

if ($suggestionsBtn) {
  $suggestionsBtn.addEventListener("click", async () => {
    $suggestionsError.innerHTML = "";
    const horizon = $suggestionsHorizon.value;
    $suggestionsBtn.innerHTML = `<span class="loading"></span>Scanning…`;
    $suggestionsBtn.disabled = true;
    $suggestionsGrid.style.display = "none";

    try {
      const res = await fetch(API_BASE + `/suggestions?limit=6&horizon=${horizon}`);
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();
      renderSuggestions(data.suggestions, data.horizon_days);
      $suggestionsBtn.innerHTML = "Refresh";
    } catch (e) {
      $suggestionsError.innerHTML = `<div class="error">${e.message || "Something went wrong."}</div>`;
      $suggestionsBtn.innerHTML = "Scan market";
    } finally {
      $suggestionsBtn.disabled = false;
    }
  });
}

function renderSuggestions(suggestions, horizonDays) {
  if (!suggestions.length) {
    $suggestionsGrid.innerHTML = `<div class="empty">No suggestions available.</div>`;
    $suggestionsGrid.style.display = "block";
    return;
  }
  $suggestionsGrid.innerHTML = `
    <div style="grid-column: 1 / -1; margin-bottom: 8px; font-size: 12px; color: var(--ink-fade);">
      Predictions for ${horizonDays}-day horizon
    </div>
  ` + suggestions.map(s => `
    <div class="suggestion-card">
      <div class="s-ticker">${s.ticker}</div>
      <div class="s-return ${signClass(parseFloat(s.predicted_return))}">
        ${s.predicted_return}
      </div>
      <div class="s-rec ${signClass(parseFloat(s.predicted_return))}">
        ${s.recommendation}
      </div>
    </div>
  `).join("");
  $suggestionsGrid.style.display = "grid";
}

// ===================================================
//   FORMATTING HELPERS
// ===================================================
function fmt(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString("en-US", {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  });
}
function fmtPct(n) {
  if (n === null || n === undefined) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${Number(n).toFixed(2)}%`;
}
function fmtSignedDollars(n) {
  if (n === null || n === undefined) return "—";
  const sign = n > 0 ? "+" : n < 0 ? "−" : "";
  return `${sign}$${fmt(Math.abs(n))}`;
}
function signClass(n) {
  if (n === null || n === undefined || n === 0) return "";
  return n > 0 ? "pos" : "neg";
}
function splitPrice(n) {
  if (n === null || n === undefined) return "—";
  const [whole, cents] = Number(n).toFixed(2).split(".");
  const wholeFmt = Number(whole).toLocaleString("en-US");
  return `$${wholeFmt}<span class="cents">.${cents}</span>`;
}