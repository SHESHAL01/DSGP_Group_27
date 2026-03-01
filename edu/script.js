// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    initializeInteractions();
    initializeScrollAnimations();
});

// Initialize the alignment chart with dark theme
function initializeChart() {
    const ctx = document.getElementById('alignmentChart');
    
    if (!ctx) return;
    
    // Chart data
    const data = {
        labels: [
            'University A (Current)',
            'Market Benchmark',
            'Competitor B',
            'Competitor C'
        ],
        datasets: [{
            label: 'Alignment %',
            data: [72, 85, 68, 60],
            backgroundColor: [
                'rgba(100, 116, 139, 0.7)',
                'rgba(45, 212, 191, 0.6)',
                'rgba(100, 116, 139, 0.7)',
                'rgba(139, 155, 179, 0.6)'
            ],
            borderColor: [
                'rgba(100, 116, 139, 1)',
                'rgba(45, 212, 191, 1)',
                'rgba(100, 116, 139, 1)',
                'rgba(139, 155, 179, 1)'
            ],
            borderWidth: 2,
            borderRadius: 10,
            barThickness: 80,
            hoverBackgroundColor: [
                'rgba(100, 116, 139, 0.9)',
                'rgba(45, 212, 191, 0.8)',
                'rgba(100, 116, 139, 0.9)',
                'rgba(139, 155, 179, 0.8)'
            ]
        }]
    };

    // Chart configuration
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 2.5,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 25, 41, 0.95)',
                    titleColor: '#2dd4bf',
                    bodyColor: '#94a3b8',
                    padding: 16,
                    borderColor: 'rgba(45, 212, 191, 0.3)',
                    borderWidth: 1,
                    displayColors: false,
                    titleFont: {
                        family: 'Poppins',
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        family: 'Poppins',
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return 'Market Alignment: ' + context.parsed.y + '%';
                        }
                    },
                    cornerRadius: 8
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            family: 'Poppins',
                            size: 12
                        },
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    title: {
                        display: true,
                        text: 'Alignment %',
                        font: {
                            family: 'Poppins',
                            size: 13,
                            weight: '600'
                        },
                        color: '#2dd4bf',
                        padding: {
                            bottom: 10
                        }
                    }
                },
                x: {
                    ticks: {
                        font: {
                            family: 'Poppins',
                            size: 11
                        },
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart',
                delay: (context) => {
                    return context.dataIndex * 150;
                }
            }
        }
    };

    // Create the chart
    const alignmentChart = new Chart(ctx, config);
    
    // Add hover effects
    ctx.addEventListener('mousemove', function(e) {
        const points = alignmentChart.getElementsAtEventForMode(e, 'nearest', { intersect: true }, false);
        if (points.length) {
            e.target.style.cursor = 'pointer';
        } else {
            e.target.style.cursor = 'default';
        }
    });
}

// Initialize interactive elements
function initializeInteractions() {
    // Refresh Data button
    const refreshBtn = document.querySelector('.btn-secondary');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            const originalContent = this.innerHTML;
            this.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>Refreshing...';
            this.disabled = true;
            
            // Add spinning animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
            
            // Simulate refresh
            setTimeout(() => {
                this.innerHTML = originalContent;
                this.disabled = false;
                showNotification('Data refreshed successfully!', 'success');
            }, 2000);
        });
    }
    
    // Export Report button
    const exportBtn = document.querySelector('.btn-primary');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            const originalContent = this.innerHTML;
            this.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" opacity="0.3"/><path d="M12 2 A10 10 0 0 1 22 12" stroke-linecap="round" style="animation: spin 1s linear infinite;"/></svg>Exporting...';
            this.disabled = true;
            
            // Simulate export
            setTimeout(() => {
                this.innerHTML = originalContent;
                this.disabled = false;
                showNotification('Report exported successfully!', 'success');
            }, 1500);
        });
    }
    
    // Add hover effects to stat cards
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        });
    });
    
    // Add click effects to insight cards
    const insightCards = document.querySelectorAll('.insight-card');
    insightCards.forEach((card, index) => {
        card.addEventListener('click', function() {
            // Add a subtle click animation
            this.style.transform = 'scale(0.98) translateY(-6px)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
    
    // Add smooth scroll for navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const links = document.querySelectorAll('.nav-link');
            links.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Initialize scroll animations
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe sections for scroll animations
    document.querySelectorAll('.chart-section, .insights-section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
        observer.observe(section);
    });
}

// Enhanced notification system with dark theme
function showNotification(message, type = 'info') {
    // Remove existing notification if any
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    
    // Set icon based on type
    let icon = '';
    let bgColor = '';
    let borderColor = '';
    
    switch(type) {
        case 'success':
            icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
            bgColor = 'rgba(16, 185, 129, 0.15)';
            borderColor = 'rgba(16, 185, 129, 0.5)';
            break;
        case 'error':
            icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>';
            bgColor = 'rgba(239, 68, 68, 0.15)';
            borderColor = 'rgba(239, 68, 68, 0.5)';
            break;
        default:
            icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
            bgColor = 'rgba(45, 212, 191, 0.15)';
            borderColor = 'rgba(45, 212, 191, 0.5)';
    }
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="color: ${borderColor}; display: flex; align-items: center;">
                ${icon}
            </div>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        background: ${bgColor};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid ${borderColor};
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        font-family: 'Poppins', sans-serif;
        font-size: 0.9375rem;
        font-weight: 500;
        z-index: 1000;
        animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-width: 280px;
    `;
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(400px);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
    
    // Add to page
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
        setTimeout(() => {
            notification.remove();
        }, 400);
    }, 3000);
}

// Add parallax effect to background
document.addEventListener('mousemove', function(e) {
    const moveX = (e.clientX - window.innerWidth / 2) * 0.01;
    const moveY = (e.clientY - window.innerHeight / 2) * 0.01;
    
    document.body.style.backgroundPosition = `${50 + moveX}% ${50 + moveY}%`;
});

// Add smooth reveal animation on page load
window.addEventListener('load', function() {
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease-out';
        document.body.style.opacity = '1';
    }, 100);
});