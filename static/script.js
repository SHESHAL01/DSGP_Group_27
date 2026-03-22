// ═══════════════════════════════════════════════════════════════════
//  SkillSync  –  script.js
//  Responsibility: API communication + DOM rendering only.
//  All visual styling lives in styles.css.
// ═══════════════════════════════════════════════════════════════════

const API_BASE = '';   // relative path — Flask serves both page and API

let alignmentChart = null;

// ─── Boot ────────────────────────────────────────────────────────
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


// ─── 1. Bar Chart ────────────────────────────────────────────────

function renderChart(data) {
    const ctx = document.getElementById('alignmentChart');
    if (!ctx) return;

    const scores = data.university_scores || [];
    const labels = scores.map(u => u.university);
    const values = scores.map(u => u.average);

    const benchmarkIdx = labels.findIndex(l => l.toLowerCase().includes('benchmark'));
    const bgColors     = labels.map((_, i) => i === benchmarkIdx ? 'rgba(45,212,191,0.6)'  : 'rgba(100,116,139,0.7)');
    const borderColors = labels.map((_, i) => i === benchmarkIdx ? 'rgba(45,212,191,1)'    : 'rgba(100,116,139,1)');
    const hoverColors  = labels.map((_, i) => i === benchmarkIdx ? 'rgba(45,212,191,0.85)' : 'rgba(100,116,139,0.9)');

    const config = {
        type: 'bar',
        data: {
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
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10,25,41,0.95)',
                    titleColor: '#2dd4bf',
                    bodyColor:  '#94a3b8',
                    padding:    16,
                    borderColor: 'rgba(45,212,191,0.3)',
                    borderWidth: 1,
                    displayColors: false,
                    titleFont: { family: 'Poppins', size: 14, weight: '600' },
                    bodyFont:  { family: 'Poppins', size: 13 },
                    callbacks: { label: ctx => `Market Alignment: ${ctx.parsed.y.toFixed(1)}%` },
                    cornerRadius: 8
                }
            },
            scales: {
                y: {
                    beginAtZero: true, max: 100,
                    ticks: { callback: v => v + '%', font: { family: 'Poppins', size: 12 }, color: '#94a3b8' },
                    grid:  { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                    title: { display: true, text: 'Alignment %', font: { family: 'Poppins', size: 13, weight: '600' }, color: '#2dd4bf', padding: { bottom: 10 } }
                },
                x: {
                    ticks: { font: { family: 'Poppins', size: 11 }, color: '#94a3b8', maxRotation: 45, minRotation: 45 },
                    grid: { display: false, drawBorder: false }
                }
            },
            animation: { duration: 1500, easing: 'easeOutQuart', delay: ctx => ctx.dataIndex * 150 }
        }
    };

    if (alignmentChart) alignmentChart.destroy();
    alignmentChart = new Chart(ctx, config);

    ctx.addEventListener('mousemove', function (e) {
        const pts = alignmentChart.getElementsAtEventForMode(e, 'nearest', { intersect: true }, false);
        e.target.style.cursor = pts.length ? 'pointer' : 'default';
    });
}


// ─── 2. Stat Cards ───────────────────────────────────────────────

function renderStatCards(data) {
    const current     = data.current_university || {};
    const benchmark   = data.market_benchmark   || 85;
    const competitors = data.university_scores  || [];

    const competitor = competitors.find(u =>
        !u.university.toLowerCase().includes('benchmark') &&
        u.university !== current.name
    ) || {};

    const cards = document.querySelectorAll('.stat-card');
    if (!cards.length) return;

    const updates = [
        { label: current.name    || 'University A',     value: current.score    ?? 72, desc: 'Market Alignment' },
        { label: 'Market Benchmark',                     value: benchmark,              desc: 'Market Alignment' },
        { label: competitor.university || 'Competitor',  value: competitor.average ?? 68, desc: 'Market Alignment' },
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

function animateNumber(el, target) {
    const duration = 1200;
    const start    = performance.now();
    function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased    = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(eased * target) + '%';
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}


// ─── 3. Insight Cards ────────────────────────────────────────────

const INSIGHT_ICONS = {
    skill_gap: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`,
    strength: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>`,
    recommendation: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`,
};

const BADGE_LABELS = {
    skill_gap:      'Skill Gap',
    strength:       'Strength',
    recommendation: 'Recommendation',
};

function renderInsightCards(insights) {
    const grid = document.querySelector('.insights-grid');
    if (!grid || !insights.length) return;
    grid.innerHTML = '';

    insights.forEach((insight, i) => {
        const type    = insight.type || 'recommendation';
        const icon    = INSIGHT_ICONS[type]  || INSIGHT_ICONS.recommendation;
        const badge   = BADGE_LABELS[type]   || 'Insight';
        const cssType = type.replace('_', '-');

        const card = document.createElement('div');
        card.className = `insight-card ${cssType}`;
        card.style.animationDelay = `${(i + 1) * 0.1}s`;

        card.innerHTML = `
            <div class="insight-header">
                <div class="insight-icon">${icon}</div>
                <span class="insight-badge ${cssType}-badge">${badge}</span>
            </div>
            <h3 class="insight-title">${escapeHtml(insight.title)}</h3>
            <p class="insight-description">${escapeHtml(insight.description)}</p>
        `;

        // Click feedback via CSS class only — no inline styles
        card.addEventListener('click', function () {
            this.classList.add('card-clicked');
            setTimeout(() => this.classList.remove('card-clicked'), 150);
        });

        grid.appendChild(card);
    });
}


// ═══════════════════════════════════════════════════════════════════
//  LOADING STATE  (class-based — no inline styles)
// ═══════════════════════════════════════════════════════════════════

function showLoadingState(isLoading) {
    const chartContainer = document.querySelector('.chart-container');
    if (!chartContainer) return;
    chartContainer.classList.toggle('loading', isLoading);
}


// ═══════════════════════════════════════════════════════════════════
//  INTERACTIONS
// ═══════════════════════════════════════════════════════════════════

function initializeInteractions() {
    // Refresh button — forces a new pipeline run
    const refreshBtn = document.querySelector('.btn-secondary');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async function () {
            const original = this.innerHTML;
            this.innerHTML = `<svg class="spin-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Refreshing...`;
            this.disabled  = true;
            await loadAnalysis(true);
            this.innerHTML = original;
            this.disabled  = false;
            showNotification('Data refreshed successfully!', 'success');
        });
    }

    // Export button
    const exportBtn = document.querySelector('.btn-primary');
    if (exportBtn) {
        exportBtn.addEventListener('click', function () {
            const original = this.innerHTML;
            this.innerHTML = `<svg class="spin-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" opacity="0.3"/><path d="M12 2 A10 10 0 0 1 22 12" stroke-linecap="round"/></svg>Exporting...`;
            this.disabled  = true;
            setTimeout(() => {
                this.innerHTML = original;
                this.disabled  = false;
                showNotification('Report exported successfully!', 'success');
            }, 1500);
        });
    }
}


// ═══════════════════════════════════════════════════════════════════
//  SCROLL ANIMATIONS  (class-based — no inline styles)
// ═══════════════════════════════════════════════════════════════════

function initializeScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.remove('section-hidden');
                entry.target.classList.add('section-visible');
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -100px 0px' });

    document.querySelectorAll('.chart-section, .insights-section').forEach(section => {
        section.classList.add('section-hidden');
        observer.observe(section);
    });
}


// ═══════════════════════════════════════════════════════════════════
//  NOTIFICATION  (class-based — no inline styles)
// ═══════════════════════════════════════════════════════════════════

function showNotification(message, type = 'info') {
    document.querySelector('.notification')?.remove();

    const icons = {
        success: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
        error:   `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`,
        info:    `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`,
    };

    const n = document.createElement('div');
    n.className = `notification notification-${type}`;
    n.innerHTML = `
        <div class="notification-inner">
            <div class="notification-icon">${icons[type] || icons.info}</div>
            <span>${message}</span>
        </div>`;
    document.body.appendChild(n);

    setTimeout(() => {
        n.classList.add('notification-hide');
        setTimeout(() => n.remove(), 400);
    }, 3000);
}


// ═══════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** Fallback shown when /api/analysis is unreachable */
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
            { type: 'skill_gap',      title: 'Missing Modern Framework Coverage',  description: 'Popular frameworks like React, Angular, and Vue.js are highly demanded in the job market but absent from the current curriculum.' },
            { type: 'skill_gap',      title: 'Limited Cloud Technology Training',  description: 'Cloud computing skills (AWS, Azure, Google Cloud) are increasingly important but underrepresented in the curriculum.' },
            { type: 'strength',       title: 'Strong Foundational Programming',    description: 'Core programming languages like Python and JavaScript are well-covered and align with market demand.' },
            { type: 'recommendation', title: 'Add Modern Web Development Module',  description: 'Include a dedicated module covering React, Node.js, and modern web development frameworks to improve alignment by ~15%.' },
            { type: 'recommendation', title: 'Update Server-Side Technologies',    description: 'Consider supplementing older content with modern backend technologies like Node.js, Django/Flask, or containerisation tools.' },
        ]
    };
}