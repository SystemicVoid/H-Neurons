#!/usr/bin/env bash
# scripts/lib/inhibit_suspend.sh
#
# Source this file from any local long-run wrapper to keep the host awake.
#
# Incident rationale (2026-04-30):
#   A 9h+ SIMID GPU run was suspended at 21:30 UTC despite a block-mode
#   `--what=sleep:idle` systemd-inhibit lock being held continuously. Trigger was
#   COSMIC DE auto-suspend from power settings. cosmic-idle (PID-resident,
#   uid 1000, no special caps) shells out to `systemctl suspend` on its own
#   timer; the narrow `sleep:idle` inhibit class did not cover every path that
#   reached `suspend.target`.
#
# Hardening this helper applies, in order:
#   1. Re-exec the calling script under systemd-inhibit with a comprehensive
#      --what= class set: sleep, idle, shutdown, handle-suspend-key,
#      handle-power-key, handle-lid-switch.
#   2. On the inhibited pass, write a far-future sentinel into COSMIC's
#      cosmic-idle config (suspend_on_ac_time, suspend_on_battery_time,
#      screen_off_time). cosmic-idle uses an inotify-based ConfigWatchSource
#      and re-reads on file change without restart. The literal RON value
#      `None` is ambiguous (may resolve to a default duration); a `Some(...)`
#      with a 68-year horizon is unambiguous "do not auto-suspend during this
#      job".
#   3. Register an EXIT/INT/TERM trap that atomically restores the original
#      values from a tempdir backup.
#
# Usage from a wrapper script (place near the top, before heavy work):
#
#   PROJECT_DIR="${PROJECT_DIR:-/home/hugo/Documents/Engineering/mech-interp/lab/02-h-neurons}"
#   # shellcheck source=scripts/lib/inhibit_suspend.sh
#   source "${PROJECT_DIR}/scripts/lib/inhibit_suspend.sh"
#   inhibit_suspend "<one-line why for systemd-inhibit>" "$@"
#
# Honours `DRY_RUN=1` (skips re-exec and config mutation entirely so dry-runs
# stay side-effect free).

# Guard against double-source.
if [[ -n "${_INHIBIT_SUSPEND_SOURCED:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
_INHIBIT_SUSPEND_SOURCED=1

# Public entrypoint -----------------------------------------------------------
inhibit_suspend() {
    local why="${1:-long local job}"
    shift || true

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi

    # First pass: re-exec the calling script under systemd-inhibit.
    if [[ -z "${INHIBIT_WRAPPED:-}" ]] && command -v systemd-inhibit &>/dev/null; then
        local inhibit_what
        inhibit_what="sleep:idle:shutdown:handle-suspend-key:handle-power-key:handle-lid-switch"
        # shellcheck disable=SC2093
        exec env INHIBIT_WRAPPED=1 systemd-inhibit \
            --what="${inhibit_what}" \
            --mode=block \
            --why="${why}" \
            -- bash "$0" "$@"
    fi

    # Inhibited pass: also disable COSMIC DE auto-suspend for the duration.
    _inhibit_suspend__disable_cosmic_idle
}

# Internals -------------------------------------------------------------------
_INHIBIT_SUSPEND_COSMIC_DIR="${HOME}/.config/cosmic/com.system76.CosmicIdle/v1"
_INHIBIT_SUSPEND_KEYS=(suspend_on_ac_time suspend_on_battery_time screen_off_time)
# Some(2147483647) seconds ~= 68 years. Far longer than any legitimate run.
_INHIBIT_SUSPEND_SENTINEL='Some(2147483647)'

_inhibit_suspend__disable_cosmic_idle() {
    [[ -d "${_INHIBIT_SUSPEND_COSMIC_DIR}" ]] || return 0

    local backup_dir
    backup_dir="$(mktemp -d -t cosmic-idle-backup-XXXXXX)" || return 0

    local key src
    for key in "${_INHIBIT_SUSPEND_KEYS[@]}"; do
        src="${_INHIBIT_SUSPEND_COSMIC_DIR}/${key}"
        if [[ -f "${src}" ]]; then
            cp -p "${src}" "${backup_dir}/${key}" 2>/dev/null || true
        else
            # Sentinel: original did not exist; restore-as-absent later.
            : > "${backup_dir}/${key}.absent"
        fi
        # Atomic write via temp + rename.
        printf '%s' "${_INHIBIT_SUSPEND_SENTINEL}" > "${src}.inhibit_tmp" 2>/dev/null || continue
        mv -f "${src}.inhibit_tmp" "${src}" 2>/dev/null || true
    done

    # cosmic-idle's calloop ConfigWatchSource picks up the change via inotify;
    # no signal needed. Send SIGHUP only as a defensive nudge if the binary
    # exists and is running.
    pkill -HUP -x cosmic-idle 2>/dev/null || true

    # shellcheck disable=SC2064
    trap "_inhibit_suspend__restore_cosmic_idle '${backup_dir}'" EXIT INT TERM
}

_inhibit_suspend__restore_cosmic_idle() {
    local backup_dir="$1"
    [[ -d "${backup_dir}" ]] || return 0

    local key dst
    for key in "${_INHIBIT_SUSPEND_KEYS[@]}"; do
        dst="${_INHIBIT_SUSPEND_COSMIC_DIR}/${key}"
        if [[ -f "${backup_dir}/${key}.absent" ]]; then
            rm -f "${dst}" 2>/dev/null || true
        elif [[ -f "${backup_dir}/${key}" ]]; then
            if cp -p "${backup_dir}/${key}" "${dst}.inhibit_tmp" 2>/dev/null; then
                mv -f "${dst}.inhibit_tmp" "${dst}" 2>/dev/null || true
            fi
        fi
    done

    pkill -HUP -x cosmic-idle 2>/dev/null || true
    rm -rf "${backup_dir}" 2>/dev/null || true
}
