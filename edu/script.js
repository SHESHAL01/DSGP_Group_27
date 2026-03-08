// ═══════════════════════════════════════════════════════════════════
//  SkillSync  –  script.js  (API-connected version)
//  Fetches live analysis data from Flask API (/api/analysis)
//  and renders the chart + insight cards dynamically.
// ═══════════════════════════════════════════════════════════════════

const API_BASE = 'http://localhost:5000';   // ← change if Flask runs on a different host/port

let alignmentChart = null;   // chart instance (kept for re-render on refresh)

// ─── Boot ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
    loadAnalysis();
    initializeInteractions();
    initializeScrollAnimations();
});


// ═══════════════════════════════════════════════════════════════════
//  DATA LOADING
// ═══════════════════════════════════════════════════════════════════

async function loadAnalysis(forceRefresh = false) {
    showLoadingState(true);

    try {
        const url = `${API_BASE}/api/analysis${forceRefresh ? '?refresh=true' : ''}`;
        const response = await fetch(url);

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const json = await response.json();

        if (json.status !== 'ok') throw new Error(json.message || 'API error');

        renderAll(json.data);

    } catch (err) {
        console.error('[SkillSync] API error:', err);
        showNotification('Could not load data from server. Showing defaults.', 'error');
        // Fall back to hardcoded defaults so the page still looks good
        renderAll(getDefaultData());
    } finally {
        showLoadingState(false);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  RENDER PIPELINE
// ═══════════════════════════════════════════════════════════════════

function renderAll(data) {
    renderChart(data);
    renderStatCards(data);
    renderInsightCards(data.insights || []);
}


// ─── 1. Bar Chart ────────────────────────────────────────────────────────────

function renderChart(data) {
    const ctx = document.getElementById('alignmentChart');
    if (!ctx) return;

    // Build labels + values from API data
    const scores = data.university_scores || [];
    const labels  = scores.map(u => u.university);
    const values  = scores.map(u => u.average);

    // Colour: teal for benchmark position, slate for the rest
    const benchmarkIdx = labels.findIndex(l => l.toLowerCase().includes('benchmark'));
    const bgColors = labels.map((_, i) =>
        i === benchmarkIdx
            ? 'rgba(45, 212, 191, 0.6)'
            : 'rgba(100, 116, 139, 0.7)'
    );
    const borderColors = labels.map((_, i) =>
        i === benchmarkIdx
            ? 'rgba(45, 212, 191, 1)'
            : 'rgba(100, 116, 139, 1)'
    );
    const hoverColors = labels.map((_, i) =>
        i === benchmarkIdx
            ? 'rgba(45, 212, 191, 0.85)'
            : 'rgba(100, 116, 139, 0.9)'
    );

    const chartData = {
        labels,
        datasets: [{
            label: 'Alignment %',
            data:  values,
            backgroundColor:      bgColors,
            borderColor:          borderColors,
            hoverBackgroundColor: hoverColors,
            borderWidth:   2,
            borderRadius:  10,
            barThickness:  80,
        }]
    };

    const config = {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 25, 41, 0.95)',
                    titleColor: '#2dd4bf',
                    bodyColor: '#94a3b8',
                    padding: 16,
                    borderColor: 'rgba(45, 212, 191, 0.3)',
                    borderWidth: 1,
                    displayColors: false,
                    titleFont:  { family: 'Poppins', size: 14, weight: '600' },
                    bodyFont:   { family: 'Poppins', size: 13 },
                    callbacks: {
                        label: ctx => `Market Alignment: ${ctx.parsed.y.toFixed(1)}%`
                    },
                    cornerRadius: 8
                }
            },
            scales: {
                y: {
                    beginAtZero: true, max: 100,
                    ticks: {
                        callback: v => v + '%',
                        font: { family: 'Poppins', size: 12 },
                        color: '#94a3b8'
                    },
                    grid:  { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                    title: {
                        display: true, text: 'Alignment %',
                        font: { family: 'Poppins', size: 13, weight: '600' },
                        color: '#2dd4bf',
                        padding: { bottom: 10 }
                    }
                },
                x: {
                    ticks: {
                        font: { family: 'Poppins', size: 11 },
                        color: '#94a3b8',
                        maxRotation: 45, minRotation: 45
                    },
                    grid: { display: false, drawBorder: false }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart',
                delay: ctx => ctx.dataIndex * 150
            }
        }
    };

    // Destroy previous instance before re-rendering
    if (alignmentChart) alignmentChart.destroy();
    alignmentChart = new Chart(ctx, config);

    ctx.addEventListener('mousemove', function (e) {
        const pts = alignmentChart.getElementsAtEventForMode(e, 'nearest', { intersect: true }, false);
        e.target.style.cursor = pts.length ? 'pointer' : 'default';
    });
}


// ─── 2. Stat Cards ───────────────────────────────────────────────────────────

function renderStatCards(data) {
    const current     = data.current_university || {};
    const benchmark   = data.market_benchmark   || 85;
    const competitors = data.university_scores  || [];

    // Find first entry that isn't current university or a benchmark label
    const competitor  = competitors.find(u =>
        !u.university.toLowerCase().includes('benchmark') &&
        u.university !== current.name
    ) || {};

    const cards = document.querySelectorAll('.stat-card');
    if (!cards.length) return;

    const updates = [
        { label: current.name    || 'University A (Current)', value: current.score ?? 72,        desc: 'Market Alignment' },
        { label: 'Market Benchmark',                           value: benchmark,                   desc: 'Market Alignment' },
        { label: competitor.university || 'Competitor B',      value: competitor.average ?? 68,   desc: 'Market Alignment' },
    ];

    cards.forEach((card, i) => {
        if (!updates[i]) return;
        const { label, value, desc } = updates[i];

        const labelEl = card.querySelector('.stat-label');
        const valueEl = card.querySelector('.stat-value');
        const descEl  = card.querySelector('.stat-desc');

        if (labelEl) labelEl.textContent = label.toUpperCase();
        if (valueEl) animateNumber(valueEl, value);
        if (descEl)  descEl.textContent  = desc;
    });
}

/** Smoothly counts a number up to the target value */
function animateNumber(el, target) {
    const duration = 1200;
    const start    = performance.now();
    const from     = 0;

    function tick(now) {
        const elapsed  = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased    = 1 - Math.pow(1 - progress, 3);   // ease-out-cubic
        el.textContent = Math.round(from + (target - from) * eased) + '%';
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}


// ─── 3. Insight Cards ────────────────────────────────────────────────────────

const INSIGHT_ICONS = {
    skill_gap: `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>`,
    strength: `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>`,
    recommendation: `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>`,
};

const BADGE_LABELS = {
    skill_gap:      'Skill Gap',
    strength:       'Strength',
    recommendation: 'Recommendation',
};

function renderInsightCards(insights) {
    const grid = document.querySelector('.insights-grid');
    if (!grid || !insights.length) return;

    grid.innerHTML = '';   // clear existing static cards

    insights.forEach((insight, i) => {
        const type  = insight.type || 'recommendation';
        const icon  = INSIGHT_ICONS[type]  || INSIGHT_ICONS.recommendation;
        const badge = BADGE_LABELS[type]   || 'Insight';

        const card = document.createElement('div');
        card.className = `insight-card ${type.replace('_', '-')}`;
        card.style.animationDelay = `${(i + 1) * 0.1}s`;

        card.innerHTML = `
            <div class="insight-header">
                <div class="insight-icon">${icon}</div>
                <span class="insight-badge ${type.replace('_', '-')}-badge">${badge}</span>
            </div>
            <h3 class="insight-title">${escapeHtml(insight.title)}</h3>
            <p class="insight-description">${escapeHtml(insight.description)}</p>
        `;

        card.addEventListener('click', function () {
            this.style.transform = 'scale(0.98) translateY(-6px)';
            setTimeout(() => { this.style.transform = ''; }, 150);
        });

        grid.appendChild(card);
    });
}


// ═══════════════════════════════════════════════════════════════════
//  LOADING STATE
// ═══════════════════════════════════════════════════════════════════

function showLoadingState(isLoading) {
    const chartContainer = document.querySelector('.chart-container');
    if (!chartContainer) return;

    if (isLoading) {
        chartContainer.setAttribute('data-loading', 'true');
        chartContainer.style.opacity = '0.5';
    } else {
        chartContainer.removeAttribute('data-loading');
        chartContainer.style.opacity = '1';
    }
}


// ═══════════════════════════════════════════════════════════════════
//  INTERACTIONS
// ═══════════════════════════════════════════════════════════════════

function initializeInteractions() {
    // Refresh button – forces a new pipeline run
    const refreshBtn = document.querySelector('.btn-secondary');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async function () {
            const original = this.innerHTML;
            const style = ensureSpinStyle();
            this.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 1s linear infinite"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Refreshing...`;
            this.disabled  = true;

            await loadAnalysis(true);   // force pipeline re-run

            this.innerHTML = original;
            this.disabled  = false;
            showNotification('Data refreshed successfully!', 'success');
        });
    }

    // Export button – unchanged behaviour
    const exportBtn = document.querySelector('.btn-primary');
    if (exportBtn) {
        exportBtn.addEventListener('click', function () {
            const original = this.innerHTML;
            this.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" opacity="0.3"/><path d="M12 2 A10 10 0 0 1 22 12" stroke-linecap="round" style="animation:spin 1s linear infinite;"/></svg>Exporting...`;
            this.disabled  = true;
            setTimeout(() => {
                this.innerHTML = original;
                this.disabled  = false;
                showNotification('Report exported successfully!', 'success');
            }, 1500);
        });
    }

    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('mouseenter', function () {
            this.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        });
    });
}


// ═══════════════════════════════════════════════════════════════════
//  SCROLL ANIMATIONS
// ═══════════════════════════════════════════════════════════════════

function initializeScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity   = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -100px 0px' });

    document.querySelectorAll('.chart-section, .insights-section').forEach(section => {
        section.style.opacity   = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
        observer.observe(section);
    });
}


// ═══════════════════════════════════════════════════════════════════
//  NOTIFICATION
// ═══════════════════════════════════════════════════════════════════

function showNotification(message, type = 'info') {
    document.querySelector('.notification')?.remove();

    const colors = {
        success: { bg: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.5)',  icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>' },
        error:   { bg: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.5)',   icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>' },
        info:    { bg: 'rgba(45,212,191,0.15)',  border: 'rgba(45,212,191,0.5)',  icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>' },
    };
    const c = colors[type] || colors.info;

    const n = document.createElement('div');
    n.className = 'notification';
    n.innerHTML = `<div style="display:flex;align-items:center;gap:.75rem"><div style="color:${c.border};display:flex;align-items:center">${c.icon}</div><span>${message}</span></div>`;
    n.style.cssText = `position:fixed;top:90px;right:20px;background:${c.bg};backdrop-filter:blur(20px);color:white;padding:1rem 1.5rem;border-radius:.75rem;border:1px solid ${c.border};box-shadow:0 10px 25px rgba(0,0,0,.5);font-family:'Poppins',sans-serif;font-size:.9375rem;font-weight:500;z-index:1000;animation:slideInRight .4s cubic-bezier(.4,0,.2,1);min-width:280px`;

    ensureSlideStyle();
    document.body.appendChild(n);

    setTimeout(() => {
        n.style.animation = 'slideOutRight .4s cubic-bezier(.4,0,.2,1)';
        setTimeout(() => n.remove(), 400);
    }, 3000);
}


// ═══════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════

function ensureSpinStyle() {
    if (document.getElementById('_spin_style')) return;
    const s = document.createElement('style');
    s.id = '_spin_style';
    s.textContent = '@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}';
    document.head.appendChild(s);
}

function ensureSlideStyle() {
    if (document.getElementById('_slide_style')) return;
    const s = document.createElement('style');
    s.id = '_slide_style';
    s.textContent = `
        @keyframes slideInRight  { from{transform:translateX(400px);opacity:0} to{transform:translateX(0);opacity:1} }
        @keyframes slideOutRight { from{transform:translateX(0);opacity:1} to{transform:translateX(400px);opacity:0} }
    `;
    document.head.appendChild(s);
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** Hardcoded fallback data – shown when the API is unreachable */
function getDefaultData() {
    return {
        current_university: { name: 'University A (Current)', score: 72 },
        market_benchmark:   85,
        university_scores: [
            { university: 'University A (Current)', average: 72 },
            { university: 'Market Benchmark',        average: 85 },
            { university: 'Competitor B',            average: 68 },
            { university: 'Competitor C',            average: 60 },
        ],
        insights: [
            { type: 'skill_gap',      title: 'Missing Modern Framework Coverage',   description: 'Popular frameworks like React, Angular, and Vue.js are highly demanded in the job market but are not present in the current curriculum.' },
            { type: 'skill_gap',      title: 'Limited Cloud Technology Training',   description: 'Cloud computing skills (AWS, Azure, Google Cloud) are increasingly important but underrepresented in the curriculum.' },
            { type: 'strength',       title: 'Strong Foundational Programming',     description: 'Core programming languages like Python and JavaScript are well-covered and align with market demand.' },
            { type: 'recommendation', title: 'Add Modern Web Development Module',   description: 'Include a dedicated module covering React, Node.js, and modern web development frameworks to improve alignment by approximately 15%.' },
            { type: 'recommendation', title: 'Update Server-Side Technologies',     description: 'Consider replacing or supplementing PHP content with more modern backend technologies like Node.js, Python frameworks (Django/Flask), or containerisation tools.' },
        ]
    };
}

// Parallax background effect
document.addEventListener('mousemove', function (e) {
    const mx = (e.clientX - window.innerWidth  / 2) * 0.01;
    const my = (e.clientY - window.innerHeight / 2) * 0.01;
    document.body.style.backgroundPosition = `${50 + mx}% ${50 + my}%`;
});

// Page fade-in
window.addEventListener('load', function () {
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-out';
        document.body.style.opacity    = '1';
    }, 100);
});
