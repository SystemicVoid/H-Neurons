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

// --- Intervention: Compliance vs alpha chart (two lines) ---
const interventionAlphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const antiComplianceRates = [64.2, 65.4, 66.0, 67.0, 68.2, 69.5, 70.5];
const standardRawRates = [69.1, 68.4, 68.8, 69.8, 68.6, 66.9, 63.6];

new Chart(document.getElementById('interventionChart'), {
  type: 'line',
  data: {
    labels: interventionAlphas.map(a => '\u03b1=' + a.toFixed(1)),
    datasets: [
      {
        label: 'Anti-compliance prompt',
        data: antiComplianceRates,
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
        data: standardRawRates,
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
          callback: (v) => v + '%',
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

// --- Parse failure chart (standard prompt) ---
const parseFailureCounts = [9, 11, 17, 32, 65, 105, 150];

new Chart(document.getElementById('parseFailureChart'), {
  type: 'bar',
  data: {
    labels: interventionAlphas.map(a => '\u03b1=' + a.toFixed(1)),
    datasets: [{
      label: 'Responses without letter prefix',
      data: parseFailureCounts,
      backgroundColor: parseFailureCounts.map(c => c > 50 ? 'rgba(255, 107, 107, 0.7)' : 'rgba(240, 165, 0, 0.5)'),
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

// --- Adjusted compliance chart (three lines) ---
const standardAdjustedRates = [69.7, 69.2, 70.0, 72.1, 73.4, 74.7, 74.8];

new Chart(document.getElementById('adjustedComplianceChart'), {
  type: 'line',
  data: {
    labels: interventionAlphas.map(a => '\u03b1=' + a.toFixed(1)),
    datasets: [
      {
        label: 'Anti-compliance prompt',
        data: antiComplianceRates,
        borderColor: '#4ecdc4',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 4,
        pointBackgroundColor: '#4ecdc4',
        borderWidth: 2.5,
      },
      {
        label: 'Standard (raw)',
        data: standardRawRates,
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
        data: standardAdjustedRates,
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
          callback: (v) => v + '%',
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

// --- Population: anti-compliance swing dynamics ---
// Stable counts are currently available for the anti-compliance prompt only.
const swingCompliant = [42, 54, 60, 70, 82, 95, 105]; // = total_compliant - 600
const swingResistant = swingCompliant.map(c => 138 - c);

new Chart(document.getElementById('populationChart'), {
  type: 'bar',
  data: {
    labels: interventionAlphas.map(a => '\u03b1=' + a.toFixed(1)),
    datasets: [
      {
        label: 'Always compliant',
        data: new Array(7).fill(600),
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
        data: new Array(7).fill(262),
        backgroundColor: 'rgba(127, 119, 221, 0.4)',
        borderRadius: {topLeft: 4, topRight: 4},
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
        max: 1000,
        grid: { color: 'rgba(157, 163, 196, 0.08)' },
        ticks: { font: { size: 12 } },
        border: { display: false },
        title: {
          display: true,
          text: 'Samples (n=1,000)',
          font: { size: 12 },
          color: '#9da3c4'
        }
      }
    }
  }
});
