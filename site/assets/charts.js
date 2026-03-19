// --- Chart.js config ---
const chartDefaults = {
  color: '#a09b91',
  borderColor: 'rgba(160, 155, 145, 0.12)',
  font: { family: "'Outfit', sans-serif", size: 12 }
};

Chart.defaults.color = chartDefaults.color;
Chart.defaults.font.family = chartDefaults.font.family;

const valueLabelPlugin = {
  id: 'valueLabels',
  afterDatasetsDraw(chart, _args, pluginOptions) {
    if (!pluginOptions || pluginOptions.disabled) {
      return;
    }

    const { ctx } = chart;
    const defaultColor = pluginOptions.color ?? '#ede8e0';
    const defaultFont = pluginOptions.font ?? { size: 11, weight: '600' };
    const defaultOffset = pluginOptions.offset ?? 10;

    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    chart.data.datasets.forEach((dataset, datasetIndex) => {
      if (!chart.isDatasetVisible(datasetIndex)) {
        return;
      }

      const meta = chart.getDatasetMeta(datasetIndex);

      meta.data.forEach((element, dataIndex) => {
        const rawValue = dataset.data[dataIndex];
        const numericValue = typeof rawValue === 'number' ? rawValue : Number(rawValue);

        if (!Number.isFinite(numericValue)) {
          return;
        }

        if (pluginOptions.skipZero && numericValue === 0) {
          return;
        }

        const formatter = pluginOptions.formatter ?? ((value) => String(value));
        const label = formatter(rawValue, { chart, dataset, datasetIndex, dataIndex });

        if (!label) {
          return;
        }

        const { x, y } = element.tooltipPosition();
        const isHorizontalBar =
          chart.config.type === 'bar' && chart.options.indexAxis === 'y';
        const isLine = chart.config.type === 'line';
        const offset =
          typeof pluginOptions.offset === 'function'
            ? pluginOptions.offset({ chart, dataset, datasetIndex, dataIndex })
            : defaultOffset;

        let drawX = x;
        let drawY = y - offset;
        let textAlign = 'center';

        if (isHorizontalBar) {
          drawX = x + offset;
          drawY = y;
          textAlign = 'left';
        } else if (isLine) {
          const direction =
            typeof pluginOptions.direction === 'function'
              ? pluginOptions.direction({ chart, dataset, datasetIndex, dataIndex })
              : pluginOptions.direction ?? (datasetIndex % 2 === 0 ? -1 : 1);
          drawY = y + offset * direction;
        }

        ctx.fillStyle =
          typeof pluginOptions.color === 'function'
            ? pluginOptions.color({ chart, dataset, datasetIndex, dataIndex })
            : defaultColor;
        ctx.font = `${defaultFont.weight ?? '600'} ${defaultFont.size ?? 11}px ${Chart.defaults.font.family}`;
        ctx.textAlign = textAlign;
        ctx.fillText(label, drawX, drawY);
      });
    });

    ctx.restore();
  },
};

Chart.register(valueLabelPlugin);

function renderHtml(targetId, html) {
  const node = document.getElementById(targetId);

  if (node) {
    node.innerHTML = html;
  }
}

function alphaChip(alpha, value) {
  return `<span class="chart-chip"><strong>\u03b1=${alpha.toFixed(1)}</strong> ${value}</span>`;
}

function renderSeriesGrid(targetId, cards) {
  renderHtml(
    targetId,
    cards
      .map(
        (card) => `
          <div class="chart-series-card">
            <div class="chart-series-title ${card.tone}">${card.title}</div>
            <div class="chart-chip-row">${card.chips.join('')}</div>
          </div>
        `
      )
      .join('')
  );
}

function setChartContainerHeight(canvas, desktopHeight, mobileHeight = 260) {
  if (!canvas?.parentElement) {
    return;
  }

  canvas.parentElement.classList.add('chart-shell');
  const targetHeight = window.matchMedia('(max-width: 720px)').matches
    ? mobileHeight
    : desktopHeight;

  canvas.parentElement.style.setProperty('--chart-shell-height', `${targetHeight}px`);
  canvas.parentElement.style.height = `${targetHeight}px`;
  canvas.parentElement.style.minHeight = `${targetHeight}px`;
  canvas.style.height = '100%';
}

// --- Classifier performance chart ---
const classifierDataUrl = new URL('../data/classifier_summary.json', import.meta.url);
let classifierDataPromise = null;
const classifierChartCanvas = document.getElementById('classifierChart');

function formatPercentFromRate(value) {
  return (value * 100).toFixed(1) + '%';
}

function formatPercentInterval(ci) {
  return `[${(ci.lower * 100).toFixed(1)}, ${(ci.upper * 100).toFixed(1)}]%`;
}

function formatClassifierMetricValue(metricName, value) {
  return metricName === 'auroc' ? value.toFixed(3) : formatPercentFromRate(value);
}

function formatClassifierMetricInterval(metricName, ci) {
  if (metricName === 'auroc') {
    return `[${ci.lower.toFixed(3)}, ${ci.upper.toFixed(3)}]`;
  }

  return formatPercentInterval(ci);
}

function loadClassifierData() {
  if (!classifierDataPromise) {
    classifierDataPromise = fetch(classifierDataUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load classifier summary: ${response.status}`);
        }
        return response.json();
      });
  }

  return classifierDataPromise;
}

async function initClassifierChart() {
  if (!classifierChartCanvas) {
    return;
  }

  setChartContainerHeight(classifierChartCanvas, 330, 250);

  const classifierData = await loadClassifierData();
  const metricOrder = ['accuracy', 'auroc', 'precision', 'recall', 'f1'];
  const metricLabels = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1'];

  new Chart(classifierChartCanvas, {
    type: 'bar',
    data: {
      labels: metricLabels,
      datasets: [{
        label: 'Disjoint test set',
        data: metricOrder.map((metricName) => classifierData.metrics[metricName].estimate),
        backgroundColor: [
          'rgba(126, 200, 160, 0.8)',
          'rgba(126, 200, 160, 0.65)',
          'rgba(123, 140, 222, 0.7)',
          'rgba(123, 140, 222, 0.55)',
          'rgba(123, 140, 222, 0.45)'
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
        valueLabels: {
          formatter: (value, context) => {
            const metricName = metricOrder[context.dataIndex];
            return formatClassifierMetricValue(metricName, value);
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const metricName = metricOrder[ctx.dataIndex];
              const metric = classifierData.metrics[metricName];
              return `${formatClassifierMetricValue(metricName, metric.estimate)} (95% CI ${formatClassifierMetricInterval(metricName, metric.ci)})`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 13, weight: '500' } },
          border: { color: 'rgba(160, 155, 145, 0.12)' }
        },
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(160, 155, 145, 0.08)' },
          ticks: {
            callback: (v) => Math.round(v * 100) + '%',
            font: { size: 12 }
          },
          border: { display: false }
        }
      }
    }
  });
}

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
  if (i <= 10) return 'rgba(230, 57, 70, 0.7)';
  if (i <= 20) return 'rgba(123, 140, 222, 0.7)';
  return 'rgba(126, 200, 160, 0.7)';
});

const layerChartCanvas = document.getElementById('layerChart');

if (layerChartCanvas) {
  setChartContainerHeight(layerChartCanvas, 320, 240);

  new Chart(layerChartCanvas, {
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
        valueLabels: {
          formatter: (value) => (value > 0 ? `${value}` : ''),
          skipZero: true,
          offset: 12,
        },
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
          border: { color: 'rgba(160, 155, 145, 0.12)' }
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(160, 155, 145, 0.08)' },
          ticks: { stepSize: 1, font: { size: 12 } },
          border: { display: false }
        }
      }
    }
  });
}

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

const topNeuronsChartCanvas = document.getElementById('topNeuronsChart');

if (topNeuronsChartCanvas) {
  setChartContainerHeight(topNeuronsChartCanvas, 340, 260);

  new Chart(topNeuronsChartCanvas, {
    type: 'bar',
    data: {
      labels: topNeurons.map(n => n.label),
      datasets: [{
        label: 'L1 weight',
        data: topNeurons.map(n => n.weight),
        backgroundColor: topNeurons.map((_, i) => i === 0 ? 'rgba(230, 57, 70, 0.85)' : 'rgba(123, 140, 222, 0.55)'),
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
        valueLabels: {
          formatter: (value) => value.toFixed(3),
          offset: 10,
        },
        tooltip: {
          callbacks: {
            label: (ctx) => 'Weight: ' + ctx.parsed.x.toFixed(3)
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(160, 155, 145, 0.08)' },
          ticks: { font: { size: 12 } },
          border: { display: false }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 12, family: "'IBM Plex Mono', monospace" } },
          border: { color: 'rgba(160, 155, 145, 0.12)' }
        }
      }
    }
  });
}

// --- Intervention charts from canonical site data ---
const interventionDataUrl = new URL('../data/intervention_sweep.json', import.meta.url);
let interventionDataPromise = null;

function formatAlphaLabel(alpha) {
  return '\u03b1=' + alpha.toFixed(1);
}

function compliancePercentages(points) {
  return points.map((point) => point.compliance_pct);
}

function interventionPointByAlpha(points, alpha) {
  const point = points.find((candidate) => candidate.alpha === alpha);

  if (!point) {
    throw new Error(`Missing intervention point for alpha=${alpha.toFixed(1)}`);
  }

  return point;
}

function formatPercent(value) {
  return value.toFixed(1) + '%';
}

function formatInterval(interval, digits = 1, suffix = '') {
  return `[${interval.lower.toFixed(digits)}, ${interval.upper.toFixed(digits)}]${suffix}`;
}

function formatCount(value) {
  return value.toLocaleString();
}

function formatCiStatus(ciStatus) {
  if (ciStatus === 'available') {
    return '95% CI available';
  }
  return ciStatus.replaceAll('_', ' ');
}

function formatSignedPp(value) {
  const sign = value >= 0 ? '+' : '';
  return sign + value.toFixed(1) + 'pp';
}

function setInterventionText(binding, value) {
  document.querySelectorAll(`[data-intervention-bind="${binding}"]`).forEach((node) => {
    node.textContent = value;
  });
}

function buildInterventionSummary(interventionData) {
  const antiComplianceSeries = interventionData.series.anti_compliance;
  const standardRawSeries = interventionData.series.standard_raw;
  const standardParseableSubsetSeries = interventionData.series.standard_parseable_subset;
  const standardTextRemapAlphaThree = interventionData.series.standard_text_remap.by_alpha['3.0'];
  const negativeControl = interventionData.negative_control.comparison_to_h_neurons;
  const parseFailures = interventionData.parse_failures.points;
  const antiBaseline = interventionPointByAlpha(antiComplianceSeries.points, 0.0);
  const antiAlphaThree = interventionPointByAlpha(antiComplianceSeries.points, 3.0);
  const antiEffects = antiComplianceSeries.effects;
  const standardRawBaseline = interventionPointByAlpha(standardRawSeries.points, 0.0);
  const standardParseableBaseline = interventionPointByAlpha(standardParseableSubsetSeries.points, 0.0);
  const parseAlphaZero = interventionPointByAlpha(parseFailures, 0.0);
  const parseAlphaThree = interventionPointByAlpha(parseFailures, 3.0);

  return {
    antiComplianceSeries,
    standardRawSeries,
    standardParseableSubsetSeries,
    standardTextRemapAlphaThree,
    negativeControl,
    parseFailures,
    antiBaseline,
    antiAlphaThree,
    antiEffects,
    standardRawBaseline,
    standardParseableBaseline,
    parseAlphaZero,
    parseAlphaThree,
  };
}

function hydrateInterventionSummary(summary) {
  const {
    antiComplianceSeries,
    standardParseableSubsetSeries,
    standardTextRemapAlphaThree,
    negativeControl,
    antiBaseline,
    antiAlphaThree,
    antiEffects,
    standardRawBaseline,
    standardParseableBaseline,
    parseAlphaZero,
    parseAlphaThree,
  } = summary;

  setInterventionText(
    'benchmark-detail',
    `${formatCount(antiBaseline.n_total)} counterfactual MC questions · per-point Wilson CIs + paired bootstrap sweep intervals`
  );
  setInterventionText('intervention-chart-n', `n=${formatCount(antiBaseline.n_total)} questions per α`);
  setInterventionText('intervention-chart-ci', `Effect: ${formatSignedPp(antiEffects.delta_0_to_max_pp.estimate)} (95% CI [${antiEffects.delta_0_to_max_pp.ci.lower.toFixed(1)}, ${antiEffects.delta_0_to_max_pp.ci.upper.toFixed(1)}]pp)`);
  setInterventionText('parse-failure-chart-n', `n=${formatCount(parseAlphaZero.n_total)} responses per α`);
  setInterventionText('parse-failure-chart-ci', 'Per-point Wilson CIs; sweep effect uses paired bootstrap');
  setInterventionText(
    'adjusted-chart-n',
    `n=1,000 total; parseable subset ${formatCount(interventionPointByAlpha(standardParseableSubsetSeries.points, 0.0).parseable_n)}→${formatCount(interventionPointByAlpha(standardParseableSubsetSeries.points, 3.0).parseable_n)}`
  );
  setInterventionText('adjusted-chart-ci', 'Wilson CIs on raw and conditional rates; strict answer-text remap is committed for α=3.0 only');
  setInterventionText('population-chart-n', `n=1,000 questions`);
  setInterventionText('population-chart-ci', 'Wilson CIs on always / never / swing shares');
  setInterventionText(
    'negative-control-chart-chip',
    `Random 38-neuron controls average ${negativeControl.slope_random_mean_pp_per_alpha.toFixed(2)}pp/α ${formatInterval(negativeControl.slope_random_percentile_interval, 2, 'pp/α')}; H-neurons move ${negativeControl.slope_h_pp_per_alpha.toFixed(2)}pp/α`
  );
  setInterventionText('anti-baseline-value', formatPercent(antiBaseline.compliance_pct));
  setInterventionText(
    'anti-baseline-detail',
    `α=0.0 compliance · 95% CI ${formatPercentInterval(antiBaseline.ci)}`
  );
  setInterventionText('standard-raw-baseline-value', formatPercent(standardRawBaseline.compliance_pct));
  setInterventionText(
    'standard-raw-baseline-detail',
    `α=0.0 raw · ${formatPercent(standardParseableBaseline.compliance_pct)} among parseable responses`
  );
  setInterventionText('anti-alpha-three-value', formatPercent(antiAlphaThree.compliance_pct));
  setInterventionText(
    'anti-alpha-three-detail',
    `${formatSignedPp(antiEffects.delta_0_to_max_pp.estimate)} from α=0.0→3.0 · 95% CI [${antiEffects.delta_0_to_max_pp.ci.lower.toFixed(1)}, ${antiEffects.delta_0_to_max_pp.ci.upper.toFixed(1)}]pp`
  );
  setInterventionText(
    'standard-remap-alpha-three-value',
    formatPercent(standardTextRemapAlphaThree.strict_rescored_compliance_pct)
  );
  setInterventionText(
    'standard-remap-alpha-three-detail',
    `${formatPercent(standardTextRemapAlphaThree.raw_compliance_pct)} raw MC-letter score · α=3.0 only answer-text remap recovered ${standardTextRemapAlphaThree.strict_recovered_count}/${standardTextRemapAlphaThree.parse_failures}`
  );
  setInterventionText('parse-alpha-zero-count', formatCount(parseAlphaZero.count));
  setInterventionText('parse-alpha-zero-detail', `${formatPercent(parseAlphaZero.pct)} of samples`);
  setInterventionText('parse-alpha-three-count', formatCount(parseAlphaThree.count));
  setInterventionText(
    'parse-alpha-three-detail',
    `${formatPercent(parseAlphaThree.pct)} · a ${(parseAlphaThree.count / parseAlphaZero.count).toFixed(1)}x increase`
  );
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
  const hasInterventionSummaryBindings = document.querySelector('[data-intervention-bind]');

  if (
    !interventionChartCanvas &&
    !parseFailureChartCanvas &&
    !adjustedComplianceChartCanvas &&
    !populationChartCanvas &&
    !hasInterventionSummaryBindings
  ) {
    return;
  }

  const interventionData = await loadInterventionData();
  const interventionAlphaLabels = interventionData.alphas.map(formatAlphaLabel);
  const summary = buildInterventionSummary(interventionData);
  const {
    antiComplianceSeries,
    standardRawSeries,
    standardParseableSubsetSeries,
    standardTextRemapAlphaThree,
    parseFailures,
    antiBaseline,
    antiAlphaThree,
    antiEffects,
    standardRawBaseline,
    standardParseableBaseline,
    parseAlphaZero,
    parseAlphaThree,
  } = summary;
  const antiCompliancePopulation = interventionData.population.anti_compliance;
  const swingBreakdown = antiCompliancePopulation.swing_breakdown;

  hydrateInterventionSummary(summary);
  renderSeriesGrid('interventionValueGrid', [
    {
      title: 'Anti-compliance',
      tone: 'teal',
      chips: [
        alphaChip(antiBaseline.alpha, formatPercent(antiBaseline.compliance_pct)),
        alphaChip(antiAlphaThree.alpha, formatPercent(antiAlphaThree.compliance_pct)),
        `<span class="chart-chip"><strong>Slope</strong> ${antiEffects.slope_pp_per_alpha.estimate.toFixed(1)}pp/α</span>`,
      ],
    },
    {
      title: 'Standard raw',
      tone: 'amber',
      chips: [
        alphaChip(standardRawBaseline.alpha, formatPercent(standardRawBaseline.compliance_pct)),
        alphaChip(standardTextRemapAlphaThree.alpha, formatPercent(standardTextRemapAlphaThree.raw_compliance_pct)),
        `<span class="chart-chip"><strong>Parse failures at α=3.0</strong> ${formatCount(parseAlphaThree.count)}</span>`,
      ],
    },
  ]);
  renderHtml('parseFailureValueStrip', [
    alphaChip(parseAlphaZero.alpha, `${formatCount(parseAlphaZero.count)} (${formatPercent(parseAlphaZero.pct)})`),
    alphaChip(parseAlphaThree.alpha, `${formatCount(parseAlphaThree.count)} (${formatPercent(parseAlphaThree.pct)})`),
    `<span class="chart-chip"><strong>Increase</strong> ${(parseAlphaThree.count / parseAlphaZero.count).toFixed(1)}× more unparseable responses</span>`,
  ].join(''));
  renderSeriesGrid('adjustedComplianceValueGrid', [
    {
      title: 'Anti-compliance',
      tone: 'teal',
      chips: [
        alphaChip(antiBaseline.alpha, formatPercent(antiBaseline.compliance_pct)),
        alphaChip(antiAlphaThree.alpha, formatPercent(antiAlphaThree.compliance_pct)),
      ],
    },
    {
      title: 'Standard raw',
      tone: 'amber',
      chips: [
        alphaChip(standardRawBaseline.alpha, formatPercent(standardRawBaseline.compliance_pct)),
        alphaChip(standardTextRemapAlphaThree.alpha, formatPercent(standardTextRemapAlphaThree.raw_compliance_pct)),
      ],
    },
    {
      title: 'Parseable subset',
      tone: 'teal',
      chips: [
        alphaChip(
          0.0,
          `${formatPercent(standardParseableBaseline.compliance_pct)} on n=${formatCount(standardParseableBaseline.parseable_n)}`
        ),
        alphaChip(
          3.0,
          `${formatPercent(interventionPointByAlpha(standardParseableSubsetSeries.points, 3.0).compliance_pct)} on n=${formatCount(interventionPointByAlpha(standardParseableSubsetSeries.points, 3.0).parseable_n)}`
        ),
      ],
    },
    {
      title: 'Strict remap',
      tone: 'coral',
      chips: [
        alphaChip(
          standardTextRemapAlphaThree.alpha,
          `${formatPercent(standardTextRemapAlphaThree.strict_rescored_compliance_pct)} full-pop correction`
        ),
      ],
    },
  ]);
  renderSeriesGrid('populationValueGrid', [
    {
      title: 'Fixed groups',
      tone: 'teal',
      chips: [
        `<span class="chart-chip"><strong>Always compliant</strong> ${formatCount(antiCompliancePopulation.always_compliant.count)} (${formatPercent(antiCompliancePopulation.always_compliant.pct)})</span>`,
        `<span class="chart-chip"><strong>Never compliant</strong> ${formatCount(antiCompliancePopulation.never_compliant.count)} (${formatPercent(antiCompliancePopulation.never_compliant.pct)})</span>`,
        `<span class="chart-chip"><strong>Swing pool</strong> ${formatCount(antiCompliancePopulation.swing.count)} (${formatPercent(antiCompliancePopulation.swing.pct)})</span>`,
      ],
    },
    {
      title: 'Swing by α',
      tone: 'teal',
      chips: [
        alphaChip(0.0, `${swingBreakdown[0].swing_compliant}/${swingBreakdown[0].swing_resistant} compliant/resistant`),
        alphaChip(3.0, `${swingBreakdown[swingBreakdown.length - 1].swing_compliant}/${swingBreakdown[swingBreakdown.length - 1].swing_resistant} compliant/resistant`),
      ],
    },
  ]);

  if (interventionChartCanvas) {
    setChartContainerHeight(interventionChartCanvas, 360, 270);

    new Chart(interventionChartCanvas, {
      type: 'line',
      data: {
        labels: interventionAlphaLabels,
        datasets: [
          {
            label: antiComplianceSeries.label,
            data: compliancePercentages(antiComplianceSeries.points),
            borderColor: '#7EC8A0',
            backgroundColor: 'rgba(126, 200, 160, 0.10)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: '#7EC8A0',
            pointBorderColor: '#7EC8A0',
            pointBorderWidth: 2,
            borderWidth: 2.5,
          },
          {
            label: 'Standard prompt (raw)',
            data: compliancePercentages(standardRawSeries.points),
            borderColor: '#D4A574',
            backgroundColor: 'rgba(212, 165, 116, 0.08)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: '#D4A574',
            pointBorderColor: '#D4A574',
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
          valueLabels: { disabled: true },
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
            ticks: { font: { size: 12, family: "'IBM Plex Mono', monospace" } },
            border: { color: 'rgba(160, 155, 145, 0.12)' }
          },
          y: {
            min: 60, max: 74,
            grid: { color: 'rgba(160, 155, 145, 0.08)' },
            ticks: {
              callback: (value) => value + '%',
              font: { size: 12 }
            },
            border: { display: false },
            title: {
              display: true,
              text: 'Compliance rate (%)',
              font: { size: 12 },
              color: '#a09b91'
            }
          }
        }
      }
    });
  }

  if (parseFailureChartCanvas) {
    setChartContainerHeight(parseFailureChartCanvas, 320, 250);

    const parseFailureTotals = parseFailures.map((point) => point.count);

    new Chart(parseFailureChartCanvas, {
      type: 'bar',
      data: {
        labels: interventionAlphaLabels,
        datasets: [{
          label: 'Responses without letter prefix',
          data: parseFailureTotals,
          backgroundColor: parseFailureTotals.map((count) => count > 50 ? 'rgba(230, 57, 70, 0.7)' : 'rgba(212, 165, 116, 0.5)'),
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
            ticks: { font: { size: 12, family: "'IBM Plex Mono', monospace" } },
            border: { color: 'rgba(160, 155, 145, 0.12)' }
          },
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(160, 155, 145, 0.08)' },
            ticks: { font: { size: 12 } },
            border: { display: false },
            title: {
              display: true,
              text: 'Parse failures (chosen=None)',
              font: { size: 12 },
              color: '#a09b91'
            }
          }
        }
      }
    });
  }

  if (adjustedComplianceChartCanvas) {
    setChartContainerHeight(adjustedComplianceChartCanvas, 360, 270);

    new Chart(adjustedComplianceChartCanvas, {
      type: 'line',
      data: {
        labels: interventionAlphaLabels,
        datasets: [
          {
            label: antiComplianceSeries.label,
            data: compliancePercentages(antiComplianceSeries.points),
            borderColor: '#7EC8A0',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: '#7EC8A0',
            borderWidth: 2.5,
          },
          {
            label: 'Standard (raw)',
            data: compliancePercentages(standardRawSeries.points),
            borderColor: '#D4A574',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: '#D4A574',
            borderWidth: 2,
            borderDash: [8, 4],
          },
          {
            label: 'Standard (parseable subset only)',
            data: compliancePercentages(standardParseableSubsetSeries.points),
            borderColor: '#a09b91',
            backgroundColor: 'rgba(160, 155, 145, 0.12)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: '#a09b91',
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
          valueLabels: { disabled: true },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 12, family: "'IBM Plex Mono', monospace" } },
            border: { color: 'rgba(160, 155, 145, 0.12)' }
          },
          y: {
            min: 60, max: 80,
            grid: { color: 'rgba(160, 155, 145, 0.08)' },
            ticks: {
              callback: (value) => value + '%',
              font: { size: 12 }
            },
            border: { display: false },
            title: {
              display: true,
              text: 'Compliance rate (%)',
              font: { size: 12 },
              color: '#a09b91'
            }
          }
        }
      }
    });
  }

  if (populationChartCanvas) {
    setChartContainerHeight(populationChartCanvas, 360, 270);

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
            backgroundColor: 'rgba(160, 155, 145, 0.35)',
            borderRadius: 0,
            borderSkipped: false,
          },
          {
            label: 'Swing \u2192 compliant',
            data: swingCompliant,
            backgroundColor: 'rgba(126, 200, 160, 0.7)',
            borderRadius: 0,
            borderSkipped: false,
          },
          {
            label: 'Swing \u2192 resistant',
            data: swingResistant,
            backgroundColor: 'rgba(212, 165, 116, 0.72)',
            borderRadius: 0,
            borderSkipped: false,
          },
          {
            label: 'Never compliant',
            data: new Array(swingBreakdown.length).fill(antiCompliancePopulation.never_compliant.count),
            backgroundColor: 'rgba(160, 155, 145, 0.18)',
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
          valueLabels: { disabled: true },
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
            ticks: { font: { size: 12, family: "'IBM Plex Mono', monospace" } },
            border: { color: 'rgba(160, 155, 145, 0.12)' }
          },
          y: {
            stacked: true,
            max: antiCompliancePopulation.n_total,
            grid: { color: 'rgba(160, 155, 145, 0.08)' },
            ticks: { font: { size: 12 } },
            border: { display: false },
            title: {
              display: true,
              text: `Samples (n=${antiCompliancePopulation.n_total.toLocaleString()})`,
              font: { size: 12 },
              color: '#a09b91'
            }
          }
        }
      }
    });
  }
}

// --- Swing characterization charts ---
const swingDataUrl = new URL('../data/swing_characterization.json', import.meta.url);
let swingDataPromise = null;

function loadSwingData() {
  if (!swingDataPromise) {
    swingDataPromise = fetch(swingDataUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load swing characterization data: ${response.status}`);
        }
        return response.json();
      });
  }

  return swingDataPromise;
}

async function initSwingCharts() {
  const transitionAlphaCanvas = document.getElementById('transitionAlphaChart');
  const knowledgeCanvas = document.getElementById('knowledgeChart');

  if (!transitionAlphaCanvas && !knowledgeCanvas) {
    return;
  }

  const swingData = await loadSwingData();

  if (transitionAlphaCanvas) {
    setChartContainerHeight(transitionAlphaCanvas, 320, 250);
    const histogram = swingData.transition_histogram;
    const container = transitionAlphaCanvas.closest('.chart-container');
    const rcSeries = histogram?.series ? (histogram.series['R\u2192C'] || histogram.series['R→C']) : null;
    const crSeries = histogram?.series ? (histogram.series['C\u2192R'] || histogram.series['C→R']) : null;
    if (!histogram?.alphas || !rcSeries?.counts_by_alpha || !crSeries?.counts_by_alpha) {
      if (container) {
        container.innerHTML = '<p class="subtitle">Transition histogram unavailable: exported raw transition counts were not found.</p>';
      }
    } else {
      const alphaLabels = histogram.alphas.map((a) => '\u03b1=' + a.toFixed(1));
      const rcCounts = histogram.alphas.map((alpha) => rcSeries.counts_by_alpha[alpha.toFixed(1)] ?? 0);
      const crCounts = histogram.alphas.map((alpha) => crSeries.counts_by_alpha[alpha.toFixed(1)] ?? 0);

      new Chart(transitionAlphaCanvas, {
        type: 'bar',
        data: {
          labels: alphaLabels,
          datasets: [
            {
              label: 'R\u2192C (knowledge override)',
              data: rcCounts,
              backgroundColor: 'rgba(230, 57, 70, 0.72)',
              borderRadius: 4,
              borderSkipped: false,
            },
            {
              label: 'C\u2192R (uncertainty resolution)',
              data: crCounts,
              backgroundColor: 'rgba(29, 158, 117, 0.72)',
              borderRadius: 4,
              borderSkipped: false,
            },
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
                font: { size: 12 }
              }
            },
            valueLabels: {
              formatter: (value) => value > 0 ? `${value}` : '',
              skipZero: true,
              offset: 12,
            },
            tooltip: {
              callbacks: {
                label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y + ' samples'
              }
            }
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { font: { size: 12, family: "'IBM Plex Mono', monospace" } },
              border: { color: 'rgba(160, 155, 145, 0.12)' }
            },
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(160, 155, 145, 0.08)' },
              ticks: { stepSize: 5, font: { size: 12 } },
              border: { display: false },
              title: {
                display: true,
                text: 'Samples transitioning at each \u03b1',
                font: { size: 12 },
                color: '#a09b91'
              }
            }
          }
        }
      });
    }
  }

  if (knowledgeCanvas) {
    setChartContainerHeight(knowledgeCanvas, 340, 260);

    const llm = swingData.llm_enrichment;
    if (llm && llm.knowledge_by_population) {
      const populations = Object.keys(llm.knowledge_by_population);
      const categories = new Set();
      populations.forEach((pop) => {
        Object.keys(llm.knowledge_by_population[pop]).forEach((cat) => categories.add(cat));
      });
      const categoryList = [...categories];
      const colors = {
        'WELL_KNOWN': 'rgba(126, 200, 160, 0.8)',
        'COMMON_KNOWLEDGE': 'rgba(126, 200, 160, 0.6)',
        'SPECIALIZED': 'rgba(212, 165, 116, 0.6)',
        'OBSCURE': 'rgba(230, 57, 70, 0.6)',
        'AMBIGUOUS': 'rgba(160, 155, 145, 0.4)',
        'well_known': 'rgba(126, 200, 160, 0.8)',
        'common_knowledge': 'rgba(126, 200, 160, 0.6)',
        'specialized': 'rgba(212, 165, 116, 0.6)',
        'obscure': 'rgba(230, 57, 70, 0.6)',
        'ambiguous': 'rgba(160, 155, 145, 0.4)',
      };
      const defaultColor = 'rgba(123, 140, 222, 0.5)';

      const datasets = categoryList.map((cat) => ({
        label: cat.replace(/_/g, ' '),
        data: populations.map((pop) => llm.knowledge_by_population[pop][cat] || 0),
        backgroundColor: colors[cat] || defaultColor,
        borderRadius: 4,
        borderSkipped: false,
      }));

      new Chart(knowledgeCanvas, {
        type: 'bar',
        data: {
          labels: populations.map((p) => p.replace(/_/g, ' ')),
          datasets,
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
            valueLabels: { disabled: true },
            tooltip: {
              callbacks: {
                label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y + ' samples'
              }
            }
          },
          scales: {
            x: {
              stacked: true,
              grid: { display: false },
              ticks: { font: { size: 12 } },
              border: { color: 'rgba(160, 155, 145, 0.12)' }
            },
            y: {
              stacked: true,
              beginAtZero: true,
              grid: { color: 'rgba(160, 155, 145, 0.08)' },
              ticks: { font: { size: 12 } },
              border: { display: false },
              title: {
                display: true,
                text: 'Samples by knowledge classification',
                font: { size: 12 },
                color: '#a09b91'
              }
            }
          }
        }
      });
    }
  }
}

initSwingCharts().catch((error) => {
  console.error('Failed to initialize swing characterization charts from site data.', error);
});

initClassifierChart().catch((error) => {
  console.error('Failed to initialize classifier chart from site data.', error);
});

initInterventionCharts().catch((error) => {
  console.error('Failed to initialize intervention charts from site data.', error);
});
