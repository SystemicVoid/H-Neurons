// --- Chart.js config ---
const chartDefaults = {
  color: '#9da3c4',
  borderColor: 'rgba(157, 163, 196, 0.12)',
  font: { family: "'Inter', sans-serif", size: 12 }
};

Chart.defaults.color = chartDefaults.color;
Chart.defaults.font.family = chartDefaults.font.family;

// --- Classifier performance chart ---
new Chart(document.getElementById('classifierChart'), {
  type: 'bar',
  data: {
    labels: ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1'],
    datasets: [{
      label: 'Disjoint test set',
      data: [0.765, 0.843, 0.767, 0.761, 0.764],
      backgroundColor: [
        'rgba(78, 205, 196, 0.8)',
        'rgba(78, 205, 196, 0.65)',
        'rgba(127, 119, 221, 0.7)',
        'rgba(127, 119, 221, 0.55)',
        'rgba(127, 119, 221, 0.45)'
      ],
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => (ctx.parsed.y * 100).toFixed(1) + '%'
        }
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: { size: 13, weight: '500' } },
        border: { color: 'rgba(157, 163, 196, 0.12)' }
      },
      y: {
        min: 0, max: 1,
        grid: { color: 'rgba(157, 163, 196, 0.08)' },
        ticks: {
          callback: (v) => Math.round(v * 100) + '%',
          font: { size: 12 }
        },
        border: { display: false }
      }
    }
  }
});

// --- Layer distribution chart ---
// Verified from models/gemma3_4b_classifier.pkl — actual per-layer H-Neuron counts
const layerData = [
  { layer: 0, count: 2 }, { layer: 2, count: 1 },
  { layer: 4, count: 3 }, { layer: 5, count: 4 },
  { layer: 6, count: 2 }, { layer: 7, count: 3 },
  { layer: 9, count: 1 }, { layer: 10, count: 2 },
  { layer: 12, count: 1 }, { layer: 13, count: 2 },
  { layer: 14, count: 2 }, { layer: 15, count: 2 },
  { layer: 16, count: 2 }, { layer: 20, count: 1 },
  { layer: 23, count: 1 }, { layer: 24, count: 1 },
  { layer: 25, count: 1 }, { layer: 26, count: 1 },
  { layer: 27, count: 1 }, { layer: 28, count: 1 },
  { layer: 30, count: 1 }, { layer: 31, count: 2 },
  { layer: 33, count: 1 }
];

const layerLabels = Array.from({length: 34}, (_, i) => 'L' + i);
const layerCounts = new Array(34).fill(0);
layerData.forEach(d => { layerCounts[d.layer] = d.count; });

const layerColors = layerCounts.map((_, i) => {
  if (i <= 10) return 'rgba(255, 107, 107, 0.7)';
  if (i <= 20) return 'rgba(127, 119, 221, 0.7)';
  return 'rgba(78, 205, 196, 0.7)';
});

new Chart(document.getElementById('layerChart'), {
  type: 'bar',
  data: {
    labels: layerLabels,
    datasets: [{
      label: 'H-Neurons',
      data: layerCounts,
      backgroundColor: layerColors,
      borderRadius: 4,
      borderSkipped: false,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => 'Layer ' + items[0].label.slice(1),
          label: (ctx) => ctx.parsed.y + ' H-Neuron' + (ctx.parsed.y !== 1 ? 's' : '')
        }
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: { size: 10 }, maxRotation: 0 },
        border: { color: 'rgba(157, 163, 196, 0.12)' }
      },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(157, 163, 196, 0.08)' },
        ticks: { stepSize: 1, font: { size: 12 } },
        border: { display: false }
      }
    }
  }
});

// --- Top neurons weight chart ---
// Verified from models/gemma3_4b_classifier.pkl — actual top-10 by L1 weight
const topNeurons = [
  { label: 'L20:N4288', weight: 12.169 },
  { label: 'L14:N8547', weight: 7.386 },
  { label: 'L13:N833', weight: 3.451 },
  { label: 'L5:N5227', weight: 3.337 },
  { label: 'L33:N8011', weight: 3.071 },
  { label: 'L24:N7995', weight: 2.603 },
  { label: 'L26:N1359', weight: 2.456 },
  { label: 'L9:N5580', weight: 1.824 },
  { label: 'L10:N4996', weight: 1.705 },
  { label: 'L0:N1819', weight: 1.693 }
];

new Chart(document.getElementById('topNeuronsChart'), {
  type: 'bar',
  data: {
    labels: topNeurons.map(n => n.label),
    datasets: [{
      label: 'L1 weight',
      data: topNeurons.map(n => n.weight),
      backgroundColor: topNeurons.map((_, i) => i === 0 ? 'rgba(255, 107, 107, 0.85)' : 'rgba(127, 119, 221, 0.55)'),
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => 'Weight: ' + ctx.parsed.x.toFixed(3)
        }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(157, 163, 196, 0.08)' },
        ticks: { font: { size: 12 } },
        border: { display: false }
      },
      y: {
        grid: { display: false },
        ticks: { font: { size: 12, family: "'JetBrains Mono', monospace" } },
        border: { color: 'rgba(157, 163, 196, 0.12)' }
      }
    }
  }
});

// --- Intervention charts from canonical site data ---
const interventionDataUrl = new URL('../data/intervention_sweep.json', import.meta.url);
let interventionDataPromise = null;

function formatAlphaLabel(alpha) {
  return '\u03b1=' + alpha.toFixed(1);
}

function compliancePercentages(points) {
  return points.map((point) => point.compliance_pct);
}

function loadInterventionData() {
  if (!interventionDataPromise) {
    interventionDataPromise = fetch(interventionDataUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load intervention sweep data: ${response.status}`);
        }
        return response.json();
      });
  }

  return interventionDataPromise;
}

async function initInterventionCharts() {
  const interventionChartCanvas = document.getElementById('interventionChart');
  const parseFailureChartCanvas = document.getElementById('parseFailureChart');
  const adjustedComplianceChartCanvas = document.getElementById('adjustedComplianceChart');
  const populationChartCanvas = document.getElementById('populationChart');

  if (!interventionChartCanvas && !parseFailureChartCanvas && !adjustedComplianceChartCanvas && !populationChartCanvas) {
    return;
  }

  const interventionData = await loadInterventionData();
  const interventionAlphaLabels = interventionData.alphas.map(formatAlphaLabel);
  const antiComplianceSeries = interventionData.series.anti_compliance;
  const standardRawSeries = interventionData.series.standard_raw;
  const standardParseableSubsetSeries = interventionData.series.standard_parseable_subset;
  const parseFailures = interventionData.parse_failures.points;
  const antiCompliancePopulation = interventionData.population.anti_compliance;
  const swingBreakdown = antiCompliancePopulation.swing_breakdown;

  if (interventionChartCanvas) {
    new Chart(interventionChartCanvas, {
      type: 'line',
      data: {
        labels: interventionAlphaLabels,
        datasets: [
          {
            label: antiComplianceSeries.label,
            data: compliancePercentages(antiComplianceSeries.points),
            borderColor: '#4ecdc4',
            backgroundColor: 'rgba(78, 205, 196, 0.10)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: '#4ecdc4',
            pointBorderColor: '#4ecdc4',
            pointBorderWidth: 2,
            borderWidth: 2.5,
          },
          {
            label: 'Standard prompt (raw)',
            data: compliancePercentages(standardRawSeries.points),
            borderColor: '#f0a500',
            backgroundColor: 'rgba(240, 165, 0, 0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: '#f0a500',
            pointBorderColor: '#f0a500',
            pointBorderWidth: 2,
            borderWidth: 2.5,
            borderDash: [8, 4],
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              usePointStyle: true,
              pointStyle: 'line',
              padding: 16,
              font: { size: 12 }
            }
          },
          tooltip: {
            callbacks: {
              title: (items) => items[0].label,
              label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 12, family: "'JetBrains Mono', monospace" } },
            border: { color: 'rgba(157, 163, 196, 0.12)' }
          },
          y: {
            min: 60, max: 74,
            grid: { color: 'rgba(157, 163, 196, 0.08)' },
            ticks: {
              callback: (value) => value + '%',
              font: { size: 12 }
            },
            border: { display: false },
            title: {
              display: true,
              text: 'Compliance rate (%)',
              font: { size: 12 },
              color: '#9da3c4'
            }
          }
        }
      }
    });
  }

  if (parseFailureChartCanvas) {
    const parseFailureTotals = parseFailures.map((point) => point.count);

    new Chart(parseFailureChartCanvas, {
      type: 'bar',
      data: {
        labels: interventionAlphaLabels,
        datasets: [{
          label: 'Responses without letter prefix',
          data: parseFailureTotals,
          backgroundColor: parseFailureTotals.map((count) => count > 50 ? 'rgba(255, 107, 107, 0.7)' : 'rgba(240, 165, 0, 0.5)'),
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.parsed.y + ' responses (' + (ctx.parsed.y / 10).toFixed(1) + '%)'
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 12, family: "'JetBrains Mono', monospace" } },
            border: { color: 'rgba(157, 163, 196, 0.12)' }
          },
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(157, 163, 196, 0.08)' },
            ticks: { font: { size: 12 } },
            border: { display: false },
            title: {
              display: true,
              text: 'Parse failures (chosen=None)',
              font: { size: 12 },
              color: '#9da3c4'
            }
          }
        }
      }
    });
  }

  if (adjustedComplianceChartCanvas) {
    new Chart(adjustedComplianceChartCanvas, {
      type: 'line',
      data: {
        labels: interventionAlphaLabels,
        datasets: [
          {
            label: antiComplianceSeries.label,
            data: compliancePercentages(antiComplianceSeries.points),
            borderColor: '#4ecdc4',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: '#4ecdc4',
            borderWidth: 2.5,
          },
          {
            label: 'Standard (raw)',
            data: compliancePercentages(standardRawSeries.points),
            borderColor: '#f0a500',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: '#f0a500',
            borderWidth: 2,
            borderDash: [8, 4],
          },
          {
            label: 'Standard (parseable subset only)',
            data: compliancePercentages(standardParseableSubsetSeries.points),
            borderColor: '#f0a500',
            backgroundColor: 'rgba(240, 165, 0, 0.10)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: '#f0a500',
            borderWidth: 2.5,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              usePointStyle: true,
              pointStyle: 'line',
              padding: 16,
              font: { size: 11 }
            }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 12, family: "'JetBrains Mono', monospace" } },
            border: { color: 'rgba(157, 163, 196, 0.12)' }
          },
          y: {
            min: 60, max: 80,
            grid: { color: 'rgba(157, 163, 196, 0.08)' },
            ticks: {
              callback: (value) => value + '%',
              font: { size: 12 }
            },
            border: { display: false },
            title: {
              display: true,
              text: 'Compliance rate (%)',
              font: { size: 12 },
              color: '#9da3c4'
            }
          }
        }
      }
    });
  }

  if (populationChartCanvas) {
    const swingCompliant = swingBreakdown.map((point) => point.swing_compliant);
    const swingResistant = swingBreakdown.map((point) => point.swing_resistant);

    new Chart(populationChartCanvas, {
      type: 'bar',
      data: {
        labels: interventionAlphaLabels,
        datasets: [
          {
            label: 'Always compliant',
            data: new Array(swingBreakdown.length).fill(antiCompliancePopulation.always_compliant.count),
            backgroundColor: 'rgba(255, 107, 107, 0.5)',
            borderRadius: 0,
            borderSkipped: false,
          },
          {
            label: 'Swing \u2192 compliant',
            data: swingCompliant,
            backgroundColor: 'rgba(78, 205, 196, 0.7)',
            borderRadius: 0,
            borderSkipped: false,
          },
          {
            label: 'Swing \u2192 resistant',
            data: swingResistant,
            backgroundColor: 'rgba(78, 205, 196, 0.25)',
            borderRadius: 0,
            borderSkipped: false,
          },
          {
            label: 'Never compliant',
            data: new Array(swingBreakdown.length).fill(antiCompliancePopulation.never_compliant.count),
            backgroundColor: 'rgba(127, 119, 221, 0.4)',
            borderRadius: { topLeft: 4, topRight: 4 },
            borderSkipped: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              pointStyle: 'rectRounded',
              padding: 16,
              font: { size: 11 }
            }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y
            }
          }
        },
        scales: {
          x: {
            stacked: true,
            grid: { display: false },
            ticks: { font: { size: 12, family: "'JetBrains Mono', monospace" } },
            border: { color: 'rgba(157, 163, 196, 0.12)' }
          },
          y: {
            stacked: true,
            max: antiCompliancePopulation.n_total,
            grid: { color: 'rgba(157, 163, 196, 0.08)' },
            ticks: { font: { size: 12 } },
            border: { display: false },
            title: {
              display: true,
              text: `Samples (n=${antiCompliancePopulation.n_total.toLocaleString()})`,
              font: { size: 12 },
              color: '#9da3c4'
            }
          }
        }
      }
    });
  }
}

initInterventionCharts().catch((error) => {
  console.error('Failed to initialize intervention charts from site data.', error);
});
