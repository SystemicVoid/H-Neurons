(() => {
  const sharedScriptUrl = document.currentScript?.src
    ? new URL(document.currentScript.src, window.location.href)
    : new URL('assets/shared.js', window.location.href);
  const progressBar = document.getElementById('progressBar');

  if (progressBar) {
    const updateProgress = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      progressBar.style.width = `${progress}%`;
    };

    window.addEventListener('scroll', updateProgress);
    updateProgress();
  }

  const visibilityObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -50px 0px' },
  );

  document
    .querySelectorAll('.fade-in, .reflection')
    .forEach((element) => visibilityObserver.observe(element));

  function animateCounter(element) {
    const target = parseInt(element.dataset.target, 10);
    const duration = parseInt(element.dataset.duration || '1500', 10);
    const start = performance.now();

    function update(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = Math.round(target * eased).toLocaleString();
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }

  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !entry.target.dataset.animated) {
          entry.target.dataset.animated = 'true';
          animateCounter(entry.target);
        }
      });
    },
    { threshold: 0.5 },
  );

  document.querySelectorAll('.counter').forEach((element) => counterObserver.observe(element));

  const scoreObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !entry.target.dataset.animated) {
          entry.target.dataset.animated = 'true';
          const items = entry.target.querySelectorAll('.score-item');
          items.forEach((item, index) => {
            setTimeout(() => {
              item.classList.add('artifact');
            }, index * 200);
          });
        }
      });
    },
    { threshold: 0.3 },
  );

  document.querySelectorAll('.scoreboard').forEach((element) => scoreObserver.observe(element));

  function setBoundText(attribute, binding, value) {
    document.querySelectorAll(`[${attribute}="${binding}"]`).forEach((element) => {
      element.textContent = value;
    });
  }

  function hasBinding(attribute) {
    return document.querySelector(`[${attribute}]`) !== null;
  }

  function formatRatePercent(rate, digits = 1) {
    return `${(rate * 100).toFixed(digits)}%`;
  }

  function formatPercent(value, digits = 1) {
    return `${value.toFixed(digits)}%`;
  }

  function formatRateCiText(ci, digits = 1) {
    return `95% CI ${(ci.lower * 100).toFixed(digits)}-${(ci.upper * 100).toFixed(digits)}%`;
  }

  function formatRateCiBracket(ci, digits = 1) {
    return `[${(ci.lower * 100).toFixed(digits)}, ${(ci.upper * 100).toFixed(digits)}]`;
  }

  function formatDecimal(value, digits = 3) {
    return value.toFixed(digits);
  }

  function formatDecimalCiText(ci, digits = 3) {
    return `95% CI ${ci.lower.toFixed(digits)}-${ci.upper.toFixed(digits)}`;
  }

  function formatDecimalCiBracket(ci, digits = 3) {
    return `[${ci.lower.toFixed(digits)}, ${ci.upper.toFixed(digits)}]`;
  }

  function formatPp(value, digits = 1) {
    return `${value.toFixed(digits)}pp`;
  }

  function formatPpCiText(ci, digits = 1) {
    return `95% CI ${ci.lower.toFixed(digits)}-${ci.upper.toFixed(digits)}pp`;
  }

  function formatIntervalText(interval, unit = '', digits = 1, label = '95% interval') {
    return `${label} ${interval.lower.toFixed(digits)}-${interval.upper.toFixed(digits)}${unit}`;
  }

  function formatPpPerAlpha(value, digits = 1) {
    return `${value.toFixed(digits)}pp/\u03b1`;
  }

  function hydrateClassifierSummary(summary) {
    const accuracy = summary.metrics.accuracy;
    const precision = summary.metrics.precision;
    const recall = summary.metrics.recall;
    const f1 = summary.metrics.f1;
    const auroc = summary.metrics.auroc;
    const overlapAccuracy = summary.overlap.metrics.accuracy;

    setBoundText('data-classifier-bind', 'selected-count', summary.selected_h_neurons.toLocaleString());
    setBoundText('data-classifier-bind', 'total-neurons', summary.total_ffn_neurons.toLocaleString());
    setBoundText(
      'data-classifier-bind',
      'selected-ratio',
      `${(summary.selected_ratio_per_mille / 10).toFixed(3)}%`,
    );
    setBoundText('data-classifier-bind', 'n-examples', summary.n_examples.toLocaleString());
    setBoundText('data-classifier-bind', 'accuracy-value', formatRatePercent(accuracy.estimate));
    setBoundText('data-classifier-bind', 'accuracy-ci-text', formatRateCiText(accuracy.ci));
    setBoundText('data-classifier-bind', 'accuracy-ci-bracket', formatRateCiBracket(accuracy.ci));
    setBoundText('data-classifier-bind', 'auc-value', formatDecimal(auroc.estimate));
    setBoundText('data-classifier-bind', 'auc-ci-text', formatDecimalCiText(auroc.ci));
    setBoundText('data-classifier-bind', 'precision-value', formatRatePercent(precision.estimate));
    setBoundText('data-classifier-bind', 'precision-ci-text', formatRateCiText(precision.ci));
    setBoundText('data-classifier-bind', 'recall-value', formatRatePercent(recall.estimate));
    setBoundText('data-classifier-bind', 'recall-ci-text', formatRateCiText(recall.ci));
    setBoundText(
      'data-classifier-bind',
      'f1-chip',
      `${formatRatePercent(f1.estimate)} ${formatRateCiBracket(f1.ci)}`,
    );
    setBoundText('data-classifier-bind', 'overlap-n-examples', summary.overlap.n_examples.toLocaleString());
    setBoundText('data-classifier-bind', 'overlap-accuracy-value', formatRatePercent(overlapAccuracy.estimate));
    setBoundText('data-classifier-bind', 'overlap-accuracy-ci-text', formatRateCiText(overlapAccuracy.ci));
    setBoundText('data-classifier-bind', 'overlap-accuracy-ci-bracket', formatRateCiBracket(overlapAccuracy.ci));
    setBoundText(
      'data-classifier-bind',
      'accuracy-drop-value',
      formatPp(summary.disjoint_accuracy_drop_vs_overlap_pp),
    );
    setBoundText(
      'data-classifier-bind',
      'disjoint-sampled-n',
      summary.disjoint_sampled_n.toLocaleString(),
    );
    setBoundText(
      'data-classifier-bind',
      'disjoint-missing-activations',
      summary.disjoint_missing_activations.toLocaleString(),
    );
  }

  function hydrateInterventionSummary(summary) {
    const antiEffects = summary.series.anti_compliance.effects;
    const swing = summary.population.anti_compliance.swing;
    const remap = summary.series.standard_text_remap.by_alpha['3.0'];
    const negativeControl = summary.negative_control.comparison_to_h_neurons;
    const swingEffectShare = antiEffects.delta_0_to_max_pp.estimate / swing.pct;

    setBoundText(
      'data-intervention-summary-bind',
      'anti-slope-value',
      `${antiEffects.slope_pp_per_alpha.estimate.toFixed(1)}pp/\u03b1`,
    );
    setBoundText(
      'data-intervention-summary-bind',
      'anti-slope-ci-text',
      `${formatPpCiText(antiEffects.slope_pp_per_alpha.ci)}/\u03b1`,
    );
    setBoundText(
      'data-intervention-summary-bind',
      'anti-delta-value',
      formatPp(antiEffects.delta_0_to_max_pp.estimate),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'anti-delta-ci-text',
      formatPpCiText(antiEffects.delta_0_to_max_pp.ci),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'swing-share-value',
      formatPercent(swing.pct),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'swing-share-ci-text',
      formatRateCiText(swing.ci),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'swing-share-ci-bracket',
      formatRateCiBracket(swing.ci),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'swing-count',
      swing.count.toLocaleString(),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'swing-effect-share',
      formatPercent(swingEffectShare),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'remap-value',
      formatPercent(remap.strict_rescored_compliance_pct),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'remap-ci-text',
      formatRateCiText(remap.strict_rescored_compliance_summary.ci),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'raw-alpha-three-value',
      formatPercent(remap.raw_compliance_pct),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'negative-control-h-slope-value',
      formatPpPerAlpha(negativeControl.slope_h_pp_per_alpha, 2),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'negative-control-random-slope-value',
      formatPpPerAlpha(negativeControl.slope_random_mean_pp_per_alpha, 2),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'negative-control-random-slope-ci-text',
      `${formatIntervalText(negativeControl.slope_random_percentile_interval, 'pp/\u03b1', 2)}`,
    );
    setBoundText(
      'data-intervention-summary-bind',
      'negative-control-h-alpha-three-value',
      formatPercent(negativeControl.alpha_3_h_rate_pct),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'negative-control-random-alpha-three-value',
      formatPercent(negativeControl.alpha_3_random_mean_pct),
    );
    setBoundText(
      'data-intervention-summary-bind',
      'negative-control-random-alpha-three-ci-text',
      formatIntervalText(negativeControl.alpha_3_random_percentile_interval_pct, '%', 1),
    );
  }

  function hydratePipelineSummary(summary) {
    const counts = summary.counts;
    const ratios = summary.ratios;
    const runtime = summary.runtime;

    setBoundText('data-pipeline-bind', 'sampled-questions', counts.sampled_questions.toLocaleString());
    setBoundText('data-pipeline-bind', 'all-correct', counts.all_correct.toLocaleString());
    setBoundText('data-pipeline-bind', 'all-incorrect', counts.all_incorrect.toLocaleString());
    setBoundText('data-pipeline-bind', 'mixed', counts.mixed.toLocaleString());
    setBoundText('data-pipeline-bind', 'consistent-total', counts.consistent_total.toLocaleString());
    setBoundText('data-pipeline-bind', 'consistent-rate', formatRatePercent(ratios.consistent_share, 0));
    setBoundText(
      'data-pipeline-bind',
      'answer-token-count',
      counts.extracted_answer_tokens.toLocaleString(),
    );
    setBoundText(
      'data-pipeline-bind',
      'extraction-failures',
      counts.extraction_failures.toLocaleString(),
    );
    setBoundText(
      'data-pipeline-bind',
      'train-sampled-total',
      counts.train_sampled_total.toLocaleString(),
    );
    setBoundText(
      'data-pipeline-bind',
      'disjoint-sampled-total',
      counts.disjoint_sampled_total.toLocaleString(),
    );
    setBoundText(
      'data-pipeline-bind',
      'disjoint-evaluated-total',
      counts.disjoint_evaluated_total.toLocaleString(),
    );
    setBoundText(
      'data-pipeline-bind',
      'disjoint-missing-activations',
      counts.disjoint_missing_activations.toLocaleString(),
    );
    setBoundText('data-pipeline-bind', 'run-cost', runtime.api_cost_display);
    setBoundText('data-pipeline-bind', 'wall-time', runtime.wall_time_display);
  }

  function hydrateSwingCharacterization(summary) {
    const subtypes = summary.subtypes;
    const rc = subtypes['R\u2192C'] || subtypes['R→C'];
    const cr = subtypes['C\u2192R'] || subtypes['C→R'];
    const nm = subtypes['non-monotonic'];
    const ta = summary.transition_alpha;
    const rcAlpha = ta['R\u2192C'] || ta['R→C'];
    const crAlpha = ta['C\u2192R'] || ta['C→R'];
    const rcCrTest = summary.rc_vs_cr_test;

    setBoundText('data-swing-bind', 'rc-count', rc.count.toLocaleString());
    setBoundText('data-swing-bind', 'rc-pct', formatPercent(rc.pct));
    setBoundText('data-swing-bind', 'rc-pct-display', formatPercent(rc.pct));
    setBoundText('data-swing-bind', 'rc-ci', `[${(rc.ci_95[0] * 100).toFixed(1)}, ${(rc.ci_95[1] * 100).toFixed(1)}]`);
    setBoundText('data-swing-bind', 'cr-count', cr.count.toLocaleString());
    setBoundText('data-swing-bind', 'cr-pct', formatPercent(cr.pct));
    setBoundText('data-swing-bind', 'cr-pct-display', formatPercent(cr.pct));
    setBoundText('data-swing-bind', 'cr-ci', `[${(cr.ci_95[0] * 100).toFixed(1)}, ${(cr.ci_95[1] * 100).toFixed(1)}]`);
    setBoundText('data-swing-bind', 'nm-count', nm.count.toLocaleString());
    setBoundText('data-swing-bind', 'nm-pct', formatPercent(nm.pct));
    setBoundText('data-swing-bind', 'rc-mean-alpha', rcAlpha.mean.toFixed(1));
    setBoundText('data-swing-bind', 'rc-median-alpha', rcAlpha.median.toFixed(2));
    setBoundText('data-swing-bind', 'cr-mean-alpha', crAlpha.mean.toFixed(1));
    setBoundText('data-swing-bind', 'cr-median-alpha', crAlpha.median.toFixed(1));
    setBoundText('data-swing-bind', 'rc-cr-p', `p=${rcCrTest.p.toFixed(2)}`);
    if (rcAlpha.early_share_le_1_5) {
      setBoundText('data-swing-bind', 'rc-early-pct', formatRatePercent(rcAlpha.early_share_le_1_5.estimate));
      setBoundText(
        'data-swing-bind',
        'rc-early-count',
        `${rcAlpha.early_share_le_1_5.count}/${rcAlpha.early_share_le_1_5.n_total}`,
      );
      setBoundText(
        'data-swing-bind',
        'rc-early-ci',
        formatRateCiBracket(rcAlpha.early_share_le_1_5.ci),
      );
    }
    if (crAlpha.early_share_le_1_5) {
      setBoundText('data-swing-bind', 'cr-early-pct', formatRatePercent(crAlpha.early_share_le_1_5.estimate));
      setBoundText(
        'data-swing-bind',
        'cr-early-count',
        `${crAlpha.early_share_le_1_5.count}/${crAlpha.early_share_le_1_5.n_total}`,
      );
      setBoundText(
        'data-swing-bind',
        'cr-early-ci',
        formatRateCiBracket(crAlpha.early_share_le_1_5.ci),
      );
    }

    const sp = summary.structural_proxies || {};
    if (sp.context_length) {
      setBoundText('data-swing-bind', 'context-length-p', `p=${sp.context_length.kruskal_p.toFixed(4)}`);
    }
    if (sp.word_overlap) {
      setBoundText('data-swing-bind', 'overlap-p', `p=${sp.word_overlap.kruskal_p.toFixed(2)}`);
    }
    if (sp.standard_response_length) {
      setBoundText('data-swing-bind', 'response-p', `p=${sp.standard_response_length.kruskal_p.toFixed(2)}`);
    }
    if (summary.source_datasets && summary.source_datasets.cramers_v != null) {
      setBoundText('data-swing-bind', 'source-v', `V=${summary.source_datasets.cramers_v.toFixed(2)}`);
    }

    const predictability = summary.structural_predictability;
    if (predictability?.tasks?.swing_vs_non_swing) {
      const primary = predictability.tasks.swing_vs_non_swing.all_ex_ante;
      const sourceOnly = predictability.tasks.swing_vs_non_swing.source_only;
      if (predictability.interpretation) {
        setBoundText('data-swing-bind', 'structural-headline', predictability.interpretation.headline);
        setBoundText('data-swing-bind', 'structural-subtitle', predictability.interpretation.subtitle);
        setBoundText('data-swing-bind', 'structural-insight', predictability.interpretation.insight);
      }
      setBoundText('data-swing-bind', 'structural-auroc', primary.auroc.estimate.toFixed(3));
      setBoundText(
        'data-swing-bind',
        'structural-auroc-ci',
        formatDecimalCiBracket(primary.auroc.ci, 3),
      );
      setBoundText(
        'data-swing-bind',
        'structural-auroc-p',
        `perm p=${primary.permutation_test.auroc.p_value.toFixed(3)}`,
      );
      setBoundText(
        'data-swing-bind',
        'structural-balanced-accuracy',
        primary.balanced_accuracy.estimate.toFixed(3),
      );
      setBoundText(
        'data-swing-bind',
        'structural-balanced-accuracy-ci',
        formatDecimalCiBracket(primary.balanced_accuracy.ci, 3),
      );
      setBoundText('data-swing-bind', 'structural-source-auroc', sourceOnly.auroc.estimate.toFixed(3));
    }
    if (predictability?.tasks?.r_to_c_vs_other_swing) {
      const subtypeTask = predictability.tasks.r_to_c_vs_other_swing.all_ex_ante;
      setBoundText('data-swing-bind', 'structural-subtype-auroc', subtypeTask.auroc.estimate.toFixed(3));
    }

    const llm = summary.llm_enrichment;
    if (llm) {
      const totalSamples = llm.samples ? llm.samples.length : null;
      if (totalSamples != null) {
        setBoundText('data-swing-bind', 'llm-sample-count', totalSamples.toLocaleString());
      }
      const agreement = llm.verification_agreement;
      if (agreement?.pct !== undefined) {
        setBoundText('data-swing-bind', 'llm-agreement', formatPercent(agreement.pct));
      } else if (llm.samples) {
        const withVerification = llm.samples.filter(
          (s) =>
            s.answer_agrees_with_model_alpha0 !== null &&
            s.answer_agrees_with_model_alpha0 !== undefined,
        );
        if (withVerification.length > 0) {
          const agreeCount = withVerification.filter((s) => s.answer_agrees_with_model_alpha0).length;
          const agreeRate = (agreeCount / withVerification.length) * 100;
          setBoundText('data-swing-bind', 'llm-agreement', formatPercent(agreeRate));
        }
      }
      // Compute overall mean persuasiveness
      if (llm.persuasiveness_by_population) {
        const pops = Object.values(llm.persuasiveness_by_population);
        const totalN = pops.reduce((s, p) => s + p.n, 0);
        const weightedSum = pops.reduce((s, p) => s + p.mean * p.n, 0);
        if (totalN > 0) {
          setBoundText('data-swing-bind', 'llm-persuasiveness', (weightedSum / totalN).toFixed(1));
        }
      }
    }
  }

  async function hydrateSiteSummaryBindings() {
    const classifierNeeded = hasBinding('data-classifier-bind');
    const interventionNeeded = hasBinding('data-intervention-summary-bind');
    const swingNeeded = hasBinding('data-swing-bind');
    const pipelineNeeded = hasBinding('data-pipeline-bind');

    if (!classifierNeeded && !interventionNeeded && !swingNeeded && !pipelineNeeded) {
      return;
    }

    const requests = [];

    if (classifierNeeded) {
      requests.push(
        fetch(new URL('../data/classifier_summary.json', sharedScriptUrl))
          .then((response) => {
            if (!response.ok) {
              throw new Error(`Failed to load classifier summary: ${response.status}`);
            }
            return response.json();
          })
          .then((summary) => hydrateClassifierSummary(summary)),
      );
    }

    if (interventionNeeded) {
      requests.push(
        fetch(new URL('../data/intervention_sweep.json', sharedScriptUrl))
          .then((response) => {
            if (!response.ok) {
              throw new Error(`Failed to load intervention summary: ${response.status}`);
            }
            return response.json();
          })
          .then((summary) => hydrateInterventionSummary(summary)),
      );
    }

    if (swingNeeded) {
      requests.push(
        fetch(new URL('../data/swing_characterization.json', sharedScriptUrl))
          .then((response) => {
            if (!response.ok) {
              throw new Error(`Failed to load swing characterization data: ${response.status}`);
            }
            return response.json();
          })
          .then((summary) => hydrateSwingCharacterization(summary)),
      );
    }

    if (pipelineNeeded) {
      requests.push(
        fetch(new URL('../data/pipeline_summary.json', sharedScriptUrl))
          .then((response) => {
            if (!response.ok) {
              throw new Error(`Failed to load pipeline summary: ${response.status}`);
            }
            return response.json();
          })
          .then((summary) => hydratePipelineSummary(summary)),
      );
    }

    await Promise.all(requests);
  }

  hydrateSiteSummaryBindings().catch((error) => {
    console.error('Failed to hydrate shared site summary bindings.', error);
  });
})();
