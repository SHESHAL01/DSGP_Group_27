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

    const scores  = data.university_scores || [];
    const labels  = scores.map(u => u.university);
    const cosineAvailable = data.cosine_available;

    const jaccardValues = scores.map(u => u.jaccard);
    const cosineValues  = scores.map(u => u.cosine);

    // Build datasets — always show Jaccard; show Cosine only if model ran
    const datasets = [];

    if (cosineAvailable) {
        datasets.push({
            label:                'Cosine Relevance %',
            data:                 cosineValues,
            backgroundColor:      'rgba(45, 212, 191, 0.7)',
            borderColor:          'rgba(45, 212, 191, 1)',
            hoverBackgroundColor: 'rgba(45, 212, 191, 0.9)',
            borderWidth:   2,
            borderRadius:  8,
            barPercentage: 0.4,
        });
    }

    datasets.push({
        label:                'Jaccard Relevance %',
        data:                 jaccardValues,
        backgroundColor:      'rgba(245, 158, 11, 0.7)',
        borderColor:          'rgba(245, 158, 11, 1)',
        hoverBackgroundColor: 'rgba(245, 158, 11, 0.9)',
        borderWidth:   2,
        borderRadius:  8,
        barPercentage: 0.4,
    });

    const config = {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color:     '#94a3b8',
                        font:      { family: 'Poppins', size: 12, weight: '500' },
                        boxWidth:  14,
                        boxHeight: 14,
                        borderRadius: 4,
                        padding:   20,
                        usePointStyle: false,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 25, 41, 0.95)',
                    titleColor:  '#2dd4bf',
                    bodyColor:   '#94a3b8',
                    padding:     16,
                    borderColor: 'rgba(45, 212, 191, 0.3)',
                    borderWidth: 1,
                    displayColors: true,
                    titleFont: { family: 'Poppins', size: 14, weight: '600' },
                    bodyFont:  { family: 'Poppins', size: 13 },
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`
                    },
                    cornerRadius: 8
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: v => v + '%',
                        font:  { family: 'Poppins', size: 12 },
                        color: '#94a3b8',
                    },
                    grid:  { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                    title: {
                        display: true,
                        text:    'Alignment Score (%)',
                        font:    { family: 'Poppins', size: 13, weight: '600' },
                        color:   '#2dd4bf',
                        padding: { bottom: 10 }
                    }
                },
                x: {
                    ticks: {
                        font:        { family: 'Poppins', size: 11 },
                        color:       '#94a3b8',
                        maxRotation: 0,
                        minRotation: 0,
                    },
                    grid: { display: false, drawBorder: false },
                    title: {
                        display: true,
                        text:    'University',
                        font:    { family: 'Poppins', size: 13, weight: '600' },
                        color:   '#94a3b8',
                        padding: { top: 10 }
                    }
                }
            },
            animation: {
                duration: 1500,
                easing:   'easeOutQuart',
                delay:    ctx => ctx.dataIndex * 100
            }
        }
    };

    if (alignmentChart) alignmentChart.destroy();
    alignmentChart = new Chart(ctx, config);

    ctx.addEventListener('mousemove', function (e) {
        const pts = alignmentChart.getElementsAtEventForMode(
            e, 'nearest', { intersect: true }, false
        );
        e.target.style.cursor = pts.length ? 'pointer' : 'default';
    });
}

// ─── 2. Stat Cards ───────────────────────────────────────────────

function renderStatCards(data) {
    const current     = data.current_university || {};
    const benchmark   = data.market_benchmark   || 0;
    const scores      = data.university_scores  || [];
    const cosineAvail = data.cosine_available;

    // Pick a university that isn't the top one for the third card
    const other = scores.find(u => u.university !== current.name) || {};

    const cards = document.querySelectorAll('.stat-card');
    if (!cards.length) return;

    const updates = [
        {
            label: (current.name || 'Top University').toUpperCase(),
            value: cosineAvail ? current.cosine : current.jaccard,
            desc:  cosineAvail ? 'Cosine Relevance' : 'Jaccard Relevance',
        },
        {
            label: 'MARKET BENCHMARK',
            value: benchmark,
            desc:  'Average of top-2 universities',
        },
        {
            label: (other.university || 'University').toUpperCase(),
            value: cosineAvail ? (other.cosine ?? other.jaccard) : other.jaccard,
            desc:  cosineAvail ? 'Cosine Relevance' : 'Jaccard Relevance',
        },
    ];

    cards.forEach((card, i) => {
        if (!updates[i]) return;
        const { label, value, desc } = updates[i];
        const labelEl = card.querySelector('.stat-label');
        const valueEl = card.querySelector('.stat-value');
        const descEl  = card.querySelector('.stat-desc');
        if (labelEl) labelEl.textContent = label;
        if (valueEl) animateNumber(valueEl, value ?? 0);
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

    // ── Build university filter bar first ─────────────────────────────
    buildUniFilter(insights);

    // ── Render cards ──────────────────────────────────────────────────
    insights.forEach((insight, i) => {
        const type    = insight.type || 'recommendation';
        const icon    = INSIGHT_ICONS[type]  || INSIGHT_ICONS.recommendation;
        const badge   = BADGE_LABELS[type]   || 'Insight';
        const cssType = type.replace('_', '-');
        const uni     = insight.university   || 'all';

        const card = document.createElement('div');
        card.className = `insight-card ${cssType}`;
        card.dataset.university = uni;
        card.style.animationDelay = `${(i + 1) * 0.08}s`;

        // Skill tags — shown only when the array is non-empty
        const skillsHtml = (insight.skills && insight.skills.length)
            ? `<div class="skill-tags">
                   ${insight.skills.map(s =>
                       `<span class="skill-tag skill-tag-${cssType}">${escapeHtml(s)}</span>`
                   ).join('')}
               </div>`
            : '';

        // Alignment score footer bar — shown only when score is present
        const scoreBadge = (insight.score !== undefined && insight.score !== null)
            ? `<div class="insight-score">
                   <span class="insight-score-label">Alignment Score</span>
                   <span class="insight-score-value">${insight.score}%</span>
               </div>`
            : '';

        // University label — shown on per-university cards only
        const uniLabel = (uni !== 'all')
            ? `<span class="uni-label">${escapeHtml(uni)}</span>`
            : '';

        card.innerHTML = `
            <div class="insight-header">
                <div class="insight-icon">${icon}</div>
                <span class="insight-badge ${cssType}-badge">${badge}</span>
                ${uniLabel}
            </div>
            <h3 class="insight-title">${escapeHtml(insight.title)}</h3>
            <p class="insight-description">${escapeHtml(insight.description)}</p>
            ${skillsHtml}
            ${scoreBadge}
        `;

        card.addEventListener('click', function () {
            this.classList.add('card-clicked');
            setTimeout(() => this.classList.remove('card-clicked'), 150);
        });

        grid.appendChild(card);
    });
}


// ─── University Filter Bar ────────────────────────────────────────

function buildUniFilter(insights) {
    const container = document.getElementById('uni-filter');
    if (!container) return;

    // Collect unique universities in appearance order; always prepend "all"
    const unis = ['all'];
    insights.forEach(ins => {
        if (ins.university && ins.university !== 'all' && !unis.includes(ins.university)) {
            unis.push(ins.university);
        }
    });

    container.innerHTML = unis.map(u => `
        <button class="filter-btn ${u === 'all' ? 'active' : ''}"
                data-filter="${escapeHtml(u)}">
            ${u === 'all' ? 'All Universities' : escapeHtml(u)}
        </button>
    `).join('');

    // Click handler — show/hide cards by university
    container.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            container.querySelectorAll('.filter-btn')
                     .forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            const selected = this.dataset.filter;
            document.querySelectorAll('.insight-card').forEach(card => {
                const cardUni = card.dataset.university;
                const visible = selected === 'all'
                    || cardUni === selected
                    || cardUni === 'all';
                card.style.display = visible ? '' : 'none';
            });
        });
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
        current_university: { name: 'University A (Current)', score: 72, jaccard: 9.5, cosine: 72 },
        market_benchmark:   70,
        cosine_available:   false,
        university_scores: [
            { university: 'University A', jaccard: 9.5,  cosine: null },
            { university: 'University B', jaccard: 7.2,  cosine: null },
            { university: 'University C', jaccard: 9.8,  cosine: null },
            { university: 'University D', jaccard: 6.1,  cosine: null },
        ],
        insights: [
            {
                type: 'skill_gap',
                university: 'University A',
                title: 'University A \u2014 Missing Skills',
                description: 'University A has a curriculum alignment score of 9.5%. The following market-demanded skills are absent or under-covered.',
                skills: ['React', 'Docker', 'Kubernetes', 'AWS', 'CI/CD'],
                score: 9.5,
            },
            {
                type: 'skill_gap',
                university: 'University B',
                title: 'University B \u2014 Missing Skills',
                description: 'University B has a curriculum alignment score of 7.2%. The following skills are underrepresented.',
                skills: ['Angular', 'Azure', 'Terraform', 'GraphQL'],
                score: 7.2,
            },
            {
                type: 'strength',
                university: 'University C',
                title: 'University C \u2014 Core Strengths',
                description: 'University C demonstrates solid curriculum coverage of the following industry-relevant skills (alignment score: 9.8%).',
                skills: ['Python', 'SQL', 'Java', 'Git', 'Agile'],
                score: 9.8,
            },
            {
                type: 'recommendation',
                university: 'all',
                title: 'Priority Skills to Add Across All Curricula',
                description: 'Average alignment across all universities is 8.2%. These skills appear as gaps in the majority of curricula.',
                skills: ['Docker', 'React', 'AWS', 'Kubernetes', 'CI/CD'],
                score: 8.2,
            },
            {
                type: 'recommendation',
                university: 'University D',
                title: 'University D \u2014 Recommended Modules',
                description: 'Introducing dedicated modules for the skills listed below could improve University D\'s alignment score by an estimated 10 percentage points.',
                skills: ['Node.js', 'MongoDB', 'TypeScript', 'Flutter'],
                score: 6.1,
            },
        ]
    };
}