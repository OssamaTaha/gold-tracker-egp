/**
 * SilverTracker EGP - Main Application
 * Premium Silver Price Tracking Dashboard for Egypt
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    API_BASE: '',  // Relative to current origin
    REFRESH_INTERVAL: 1 * 60 * 1000,  // 1 minute
    DEFAULT_PERIOD: '1m'
};

// ============================================================================
// STATE
// ============================================================================

const state = {
    chart: null,
    areaSeries: null,
    currentPeriod: CONFIG.DEFAULT_PERIOD,
    chartType: '999', // '999', '925', '900', '800', 'oz'
    isLoading: false,
    lastUpdate: null
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Format number as currency (EGP)
 */
function formatPrice(value) {
    return new Intl.NumberFormat('en-EG', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format price as USD
 */
function formatPriceUSD(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format price change with sign
 */
function formatChange(value) {
    const sign = value > 0 ? '+' : '';
    return `${sign}${formatPrice(value)}`;
}

/**
 * Fetch data from API with error handling
 */
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// ============================================================================
// PRICE CARDS
// ============================================================================

/**
 * Update a single price card
 */
function updatePriceCard(purity, data) {
    const priceEl = document.getElementById(`price${purity}`);
    const changeEl = document.getElementById(`change${purity}`);

    if (!priceEl || !changeEl) return;

    // Determine value and formatting
    const isOz = purity === 'oz';
    const value = data.value || data;
    const formattedPrice = isOz ? formatPriceUSD(value) : formatPrice(value);

    // Update price value
    priceEl.innerHTML = formattedPrice;

    // Update change indicator if available
    if (data.change !== undefined) {
        const direction = data.direction || 'neutral';
        changeEl.className = `price-change ${direction}`;
        changeEl.innerHTML = `
            <span class="change-value">${formatChange(data.change)}</span>
            <span class="change-percent">(${data.change_pct > 0 ? '+' : ''}${data.change_pct.toFixed(2)}%)</span>
        `;
    } else {
        changeEl.innerHTML = '<span class="change-value">--</span>';
    }

    // Add animation
    priceEl.closest('.price-card').classList.add('fade-in');
}

/**
 * Fetch and update all price cards
 */
async function updatePrices() {
    try {
        const data = await fetchAPI('/api/silver/prices/current');

        // Update timestamp
        state.lastUpdate = new Date(data.timestamp);
        updateLastUpdateDisplay();

        // Update each purity card
        updatePriceCard('999', data.prices.purity_999);
        updatePriceCard('925', data.prices.purity_925);
        updatePriceCard('900', data.prices.purity_900);
        updatePriceCard('800', data.prices.purity_800);

        // Update Ounce card
        if (data.silver_usd_oz) {
            updatePriceCard('oz', data.silver_usd_oz);
        }

    } catch (error) {
        console.error('Failed to update prices:', error);
        showPriceError();
    }
}

/**
 * Show error state for prices
 */
function showPriceError() {
    ['999', '925', '900', '800', 'oz'].forEach(purity => {
        const priceEl = document.getElementById(`price${purity}`);
        if (priceEl) {
            priceEl.innerHTML = '--';
        }
    });
}

/**
 * Update the last update display
 */
function updateLastUpdateDisplay() {
    const el = document.getElementById('lastUpdate');
    if (el && state.lastUpdate) {
        const time = state.lastUpdate.toLocaleTimeString('en-EG', {
            hour: '2-digit',
            minute: '2-digit'
        });
        el.innerHTML = `
            <span class="pulse"></span>
            <span>Live • ${time}</span>
        `;
    }
}

// ============================================================================
// AREA CHART - Price Trend
// ============================================================================

/**
 * Initialize the TradingView Lightweight Area Chart
 */
function initChart() {
    const container = document.getElementById('candlestickChart');
    if (!container) return;

    // Create chart with dark theme and silver colors
    state.chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 400,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#a0a0a0'
        },
        grid: {
            vertLines: { color: 'rgba(192, 192, 192, 0.08)' },
            horzLines: { color: 'rgba(192, 192, 192, 0.08)' }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                color: 'rgba(192, 192, 192, 0.5)',
                width: 1,
                style: LightweightCharts.LineStyle.Dashed,
                labelBackgroundColor: '#C0C0C0'
            },
            horzLine: {
                color: 'rgba(192, 192, 192, 0.5)',
                width: 1,
                style: LightweightCharts.LineStyle.Dashed,
                labelBackgroundColor: '#C0C0C0'
            }
        },
        rightPriceScale: {
            borderColor: 'rgba(192, 192, 192, 0.2)',
            scaleMargins: {
                top: 0.1,
                bottom: 0.1
            }
        },
        timeScale: {
            borderColor: 'rgba(192, 192, 192, 0.2)',
            timeVisible: true,
            secondsVisible: false
        },
        handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true
        },
        handleScale: {
            axisPressedMouseMove: true,
            mouseWheel: true,
            pinch: true
        }
    });

    // Add area series with silver gradient
    state.areaSeries = state.chart.addAreaSeries({
        topColor: 'rgba(192, 192, 192, 0.4)',
        bottomColor: 'rgba(192, 192, 192, 0.0)',
        lineColor: '#C0C0C0',
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 6,
        crosshairMarkerBorderColor: '#C0C0C0',
        crosshairMarkerBackgroundColor: '#0a0a0a'
    });

    // Create price legend
    createPriceLegend(container);

    // Subscribe to crosshair move
    state.chart.subscribeCrosshairMove(updatePriceLegend);

    // Handle resize
    window.addEventListener('resize', () => {
        if (state.chart) {
            state.chart.applyOptions({
                width: container.clientWidth
            });
        }
    });
}

/**
 * Create Price Legend element
 */
function createPriceLegend(container) {
    const legend = document.createElement('div');
    legend.id = 'priceLegend';
    legend.style.cssText = `
        position: absolute;
        top: 12px;
        left: 12px;
        z-index: 10;
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        pointer-events: none;
    `;
    legend.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <span id="legend-title" style="color: #C0C0C0; font-weight: 600; font-size: 14px;">Silver Price (999)</span>
            <span id="legend-currency" style="color: #666; font-size: 12px;">EGP/Gram</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 12px;">
            <span id="legend-price" style="color: #fff; font-size: 24px; font-weight: 700;">--</span>
            <span id="legend-date" style="color: #666; font-size: 12px;">--</span>
        </div>
    `;
    container.style.position = 'relative';
    container.appendChild(legend);
}

/**
 * Update Price Legend on crosshair move
 */
function updatePriceLegend(param) {
    const priceEl = document.getElementById('legend-price');
    const dateEl = document.getElementById('legend-date');
    const titleEl = document.getElementById('legend-title');
    const currEl = document.getElementById('legend-currency');

    if (state.chartType === 'oz') {
        titleEl.textContent = 'Silver Ounce (Global)';
        currEl.textContent = 'USD';
    } else {
        titleEl.textContent = `Silver Price (${state.chartType})`;
        currEl.textContent = 'EGP/Gram';
    }

    if (!priceEl || !param.time || !param.seriesData) return;

    const data = param.seriesData.get(state.areaSeries);
    if (!data) return;

    // Format price
    if (state.chartType === 'oz') {
        priceEl.textContent = formatPriceUSD(data.value);
    } else {
        priceEl.textContent = formatPrice(data.value) + ' EGP';
    }

    // Format date
    const date = new Date(param.time * 1000);
    dateEl.textContent = date.toLocaleDateString('en-EG', {
        day: '2-digit',
        month: 'short',
        year: 'numeric'
    });
}

/**
 * Fetch and update chart data
 */
async function updateChart(period = null, type = null) {
    if (period) state.currentPeriod = period;
    if (type) state.chartType = type;

    const loadingEl = document.getElementById('chartLoading');

    try {
        if (loadingEl) loadingEl.classList.remove('hidden');

        const data = await fetchAPI(`/api/silver/prices/ohlc?period=${state.currentPeriod}&type=${state.chartType}`);

        if (data.data && data.data.length > 0) {
            // Convert OHLC data to line data (use close price)
            const lineData = data.data.map(d => ({
                time: d.time,
                value: d.close
            }));
            state.areaSeries.setData(lineData);
            state.chart.timeScale().fitContent();

            // Auto scale price axis
            state.chart.priceScale('right').applyOptions({
                autoScale: true
            });

            // Update legend with latest value
            const latest = lineData[lineData.length - 1];
            const priceEl = document.getElementById('legend-price');
            if (priceEl && latest) {
                if (state.chartType === 'oz') {
                    priceEl.textContent = formatPriceUSD(latest.value);
                } else {
                    priceEl.textContent = formatPrice(latest.value) + ' EGP';
                }
            }
            // Update legend title immediately
            updatePriceLegend({ time: null, seriesData: null });

        } else {
            console.warn('No price data available');
        }

    } catch (error) {
        console.error('Failed to update chart:', error);
    } finally {
        if (loadingEl) loadingEl.classList.add('hidden');
    }
}

/**
 * Setup chart controls (Timeframe and Type)
 */
function setupChartControls() {
    // Timeframe selector
    const tfButtons = document.querySelectorAll('#timeframeSelector .timeframe-btn');
    tfButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tfButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateChart(btn.dataset.period, null);
        });
    });

    // Type selector
    const typeButtons = document.querySelectorAll('#typeSelector .timeframe-btn');
    typeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            typeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateChart(null, btn.dataset.type);
        });
    });
}

// ============================================================================
// NEWS SECTION
// ============================================================================

/**
 * Fetch and display news
 */
async function updateNews() {
    const container = document.getElementById('newsGrid');
    if (!container) return;

    try {
        const data = await fetchAPI('/api/silver/news?limit=6');

        if (data.news && data.news.length > 0) {
            container.innerHTML = data.news.map((item, index) => `
                <a href="${item.link}" target="_blank" rel="noopener noreferrer" 
                   class="news-card fade-in" style="animation-delay: ${0.1 + index * 0.05}s">
                    <h3 class="news-title">${escapeHtml(item.title)}</h3>
                    <div class="news-meta">
                        <span class="news-source-badge">${escapeHtml(item.source)}</span>
                        <span class="news-time">${item.time_ago || item.date}</span>
                    </div>
                </a>
            `).join('');
        } else {
            container.innerHTML = `
                <div class="news-loading">
                    <span>No silver news available at the moment</span>
                </div>
            `;
        }

    } catch (error) {
        console.error('Failed to fetch news:', error);
        container.innerHTML = `
            <div class="news-loading">
                <span>Failed to load news</span>
            </div>
        `;
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// REFRESH FUNCTIONALITY
// ============================================================================

/**
 * Refresh all data
 */
async function refreshAll() {
    const refreshBtn = document.getElementById('refreshBtn');

    if (state.isLoading) return;
    state.isLoading = true;

    if (refreshBtn) refreshBtn.classList.add('loading');

    try {
        await Promise.all([
            updatePrices(),
            updateChart(),
            updateNews()
        ]);
    } finally {
        state.isLoading = false;
        if (refreshBtn) refreshBtn.classList.remove('loading');
    }
}

/**
 * Setup refresh button
 */
function setupRefreshButton() {
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshAll);
    }
}

/**
 * Setup auto-refresh interval
 */
function setupAutoRefresh() {
    setInterval(() => {
        refreshAll();
    }, CONFIG.REFRESH_INTERVAL);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the application
 */
async function init() {
    console.log('🥈 SilverTracker EGP - Initializing...');

    // Initialize chart
    initChart();

    // Setup event listeners
    setupChartControls();
    setupRefreshButton();

    // Initial data load
    await refreshAll();

    // Setup auto-refresh
    setupAutoRefresh();

    console.log('✅ SilverTracker EGP - Ready!');
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
