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
#   3. Register an EXIT/INT/TERM trap that restores the original values from a
#      shared per-user backup after the last overlapping run exits.
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
    # shellcheck disable=SC2317
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
_INHIBIT_SUSPEND_STATE_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/inhibit-suspend-cosmic-idle-${UID}"
_INHIBIT_SUSPEND_ACTIVE_DIR="${_INHIBIT_SUSPEND_STATE_DIR}/active"
_INHIBIT_SUSPEND_ORIGINAL_DIR="${_INHIBIT_SUSPEND_STATE_DIR}/original"
_INHIBIT_SUSPEND_COSMIC_TOKEN=
_INHIBIT_SUSPEND_PREV_TRAP_EXIT_SET=0
_INHIBIT_SUSPEND_PREV_TRAP_EXIT=
_INHIBIT_SUSPEND_PREV_TRAP_INT_SET=0
_INHIBIT_SUSPEND_PREV_TRAP_INT=
_INHIBIT_SUSPEND_PREV_TRAP_TERM_SET=0
_INHIBIT_SUSPEND_PREV_TRAP_TERM=

_inhibit_suspend__disable_cosmic_idle() {
    [[ -d "${_INHIBIT_SUSPEND_COSMIC_DIR}" ]] || return 0
    [[ -z "${_INHIBIT_SUSPEND_COSMIC_TOKEN:-}" ]] || return 0
    command -v flock &>/dev/null || return 0

    _INHIBIT_SUSPEND_COSMIC_TOKEN="${BASHPID}.${RANDOM}.${RANDOM}"
    if ! _inhibit_suspend__with_cosmic_lock _inhibit_suspend__activate_cosmic_idle_locked; then
        _INHIBIT_SUSPEND_COSMIC_TOKEN=
        return 0
    fi

    _inhibit_suspend__install_traps
}

_inhibit_suspend__with_cosmic_lock() {
    mkdir -p "${_INHIBIT_SUSPEND_STATE_DIR}" || return 1
    chmod 700 "${_INHIBIT_SUSPEND_STATE_DIR}" 2>/dev/null || true

    local lock_fd
    exec {lock_fd}>"${_INHIBIT_SUSPEND_STATE_DIR}/lock" || return 1
    flock "${lock_fd}" || {
        exec {lock_fd}>&-
        return 1
    }

    "$@"
    local status=$?
    exec {lock_fd}>&-
    return "${status}"
}

_inhibit_suspend__activate_cosmic_idle_locked() {
    mkdir -p "${_INHIBIT_SUSPEND_ACTIVE_DIR}" || return 1
    chmod 700 "${_INHIBIT_SUSPEND_ACTIVE_DIR}" 2>/dev/null || true
    _inhibit_suspend__prune_dead_active_tokens

    if ! _inhibit_suspend__has_active_tokens; then
        if [[ ! -d "${_INHIBIT_SUSPEND_ORIGINAL_DIR}" ]] || ! _inhibit_suspend__cosmic_idle_has_sentinel; then
            _inhibit_suspend__capture_cosmic_idle_originals || return 1
        fi
    fi

    printf '%s\n' "$$" > "${_INHIBIT_SUSPEND_ACTIVE_DIR}/${_INHIBIT_SUSPEND_COSMIC_TOKEN}" || return 1
    _inhibit_suspend__write_cosmic_idle_sentinel
}

_inhibit_suspend__capture_cosmic_idle_originals() {
    mkdir -p "${_INHIBIT_SUSPEND_ORIGINAL_DIR}" || return 1
    chmod 700 "${_INHIBIT_SUSPEND_ORIGINAL_DIR}" 2>/dev/null || true

    local key src
    for key in "${_INHIBIT_SUSPEND_KEYS[@]}"; do
        src="${_INHIBIT_SUSPEND_COSMIC_DIR}/${key}"
        rm -f "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}" "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}.absent" 2>/dev/null || true
        if [[ -f "${src}" ]]; then
            cp -p "${src}" "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}" 2>/dev/null || return 1
        else
            # Sentinel: original did not exist; restore-as-absent later.
            : > "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}.absent" || return 1
        fi
    done
}

_inhibit_suspend__write_cosmic_idle_sentinel() {
    local key dst tmp
    for key in "${_INHIBIT_SUSPEND_KEYS[@]}"; do
        dst="${_INHIBIT_SUSPEND_COSMIC_DIR}/${key}"
        tmp="${dst}.inhibit_tmp.${_INHIBIT_SUSPEND_COSMIC_TOKEN}"
        if printf '%s' "${_INHIBIT_SUSPEND_SENTINEL}" > "${tmp}" 2>/dev/null; then
            mv -f "${tmp}" "${dst}" 2>/dev/null || true
        else
            rm -f "${tmp}" 2>/dev/null || true
        fi
    done
}

_inhibit_suspend__cosmic_idle_has_sentinel() {
    local key src value
    for key in "${_INHIBIT_SUSPEND_KEYS[@]}"; do
        src="${_INHIBIT_SUSPEND_COSMIC_DIR}/${key}"
        [[ -f "${src}" ]] || return 1
        value="$(<"${src}")"
        [[ "${value}" == "${_INHIBIT_SUSPEND_SENTINEL}" ]] || return 1
    done
}

_inhibit_suspend__restore_cosmic_idle() {
    [[ -n "${_INHIBIT_SUSPEND_COSMIC_TOKEN:-}" ]] || return 0
    _inhibit_suspend__with_cosmic_lock _inhibit_suspend__restore_cosmic_idle_locked || true
    _INHIBIT_SUSPEND_COSMIC_TOKEN=
}

_inhibit_suspend__restore_cosmic_idle_locked() {
    rm -f "${_INHIBIT_SUSPEND_ACTIVE_DIR}/${_INHIBIT_SUSPEND_COSMIC_TOKEN}" 2>/dev/null || true
    _inhibit_suspend__prune_dead_active_tokens
    _inhibit_suspend__has_active_tokens && return 0

    [[ -d "${_INHIBIT_SUSPEND_ORIGINAL_DIR}" ]] || return 0

    local key dst
    for key in "${_INHIBIT_SUSPEND_KEYS[@]}"; do
        dst="${_INHIBIT_SUSPEND_COSMIC_DIR}/${key}"
        if [[ -f "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}.absent" ]]; then
            rm -f "${dst}" 2>/dev/null || true
        elif [[ -f "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}" ]]; then
            if cp -p "${_INHIBIT_SUSPEND_ORIGINAL_DIR}/${key}" "${dst}.inhibit_tmp" 2>/dev/null; then
                mv -f "${dst}.inhibit_tmp" "${dst}" 2>/dev/null || true
            fi
        fi
    done

    rm -rf "${_INHIBIT_SUSPEND_ORIGINAL_DIR}" "${_INHIBIT_SUSPEND_ACTIVE_DIR}" 2>/dev/null || true
}

_inhibit_suspend__prune_dead_active_tokens() {
    [[ -d "${_INHIBIT_SUSPEND_ACTIVE_DIR}" ]] || return 0

    local marker pid
    while IFS= read -r -d '' marker; do
        pid="$(<"${marker}")"
        if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${marker}" 2>/dev/null || true
        fi
    done < <(find "${_INHIBIT_SUSPEND_ACTIVE_DIR}" -mindepth 1 -maxdepth 1 -type f -print0 2>/dev/null)
}

_inhibit_suspend__has_active_tokens() {
    local active_token
    IFS= read -r active_token < <(find "${_INHIBIT_SUSPEND_ACTIVE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -quit 2>/dev/null)
    [[ -n "${active_token}" ]]
}

_inhibit_suspend__install_traps() {
    _inhibit_suspend__capture_existing_trap EXIT
    _inhibit_suspend__capture_existing_trap INT
    _inhibit_suspend__capture_existing_trap TERM
    _inhibit_suspend__install_managed_trap EXIT
    _inhibit_suspend__install_managed_trap INT
    _inhibit_suspend__install_managed_trap TERM
}

_inhibit_suspend__capture_existing_trap() {
    local signal="$1"
    local output_signal
    output_signal="$(_inhibit_suspend__trap_output_signal "${signal}")"
    local line
    line="$(builtin trap -p "${signal}")" || line=

    local body_var set_var regex
    body_var="_INHIBIT_SUSPEND_PREV_TRAP_${signal}"
    set_var="_INHIBIT_SUSPEND_PREV_TRAP_${signal}_SET"
    regex="^trap -- '(.*)' ${output_signal}$"
    if [[ "${line}" =~ ${regex} ]]; then
        printf -v "${body_var}" '%s' "${BASH_REMATCH[1]}"
        printf -v "${set_var}" '%s' "1"
    else
        printf -v "${body_var}" '%s' ""
        printf -v "${set_var}" '%s' "0"
    fi
}

_inhibit_suspend__trap_output_signal() {
    case "$1" in
        EXIT) printf '%s\n' EXIT ;;
        INT) printf '%s\n' SIGINT ;;
        TERM) printf '%s\n' SIGTERM ;;
    esac
}

_inhibit_suspend__normalize_managed_signal() {
    case "$1" in
        EXIT | 0) printf '%s\n' EXIT ;;
        INT | SIGINT | 2) printf '%s\n' INT ;;
        TERM | SIGTERM | 15) printf '%s\n' TERM ;;
    esac
}

_inhibit_suspend__set_previous_trap() {
    local signal="$1"
    local action="$2"
    local body_var set_var
    body_var="_INHIBIT_SUSPEND_PREV_TRAP_${signal}"
    set_var="_INHIBIT_SUSPEND_PREV_TRAP_${signal}_SET"

    if [[ "${action}" == "-" ]]; then
        printf -v "${body_var}" '%s' ""
        printf -v "${set_var}" '%s' "0"
    else
        printf -v "${body_var}" '%s' "${action}"
        printf -v "${set_var}" '%s' "1"
    fi
}

_inhibit_suspend__install_managed_trap() {
    case "$1" in
        EXIT) builtin trap '_inhibit_suspend__handle_exit "$?"' EXIT ;;
        INT) builtin trap '_inhibit_suspend__handle_signal INT "$?"' INT ;;
        TERM) builtin trap '_inhibit_suspend__handle_signal TERM "$?"' TERM ;;
    esac
}

_inhibit_suspend__set_status() {
    return "$1"
}

_inhibit_suspend__handle_exit() {
    local status="$1"
    _inhibit_suspend__restore_cosmic_idle

    if [[ "${_INHIBIT_SUSPEND_PREV_TRAP_EXIT_SET}" == "1" ]]; then
        _inhibit_suspend__set_status "${status}"
        eval "${_INHIBIT_SUSPEND_PREV_TRAP_EXIT}"
    fi
    return "${status}"
}

_inhibit_suspend__handle_signal() {
    local signal="$1"
    local status="$2"
    local body_var set_var
    body_var="_INHIBIT_SUSPEND_PREV_TRAP_${signal}"
    set_var="_INHIBIT_SUSPEND_PREV_TRAP_${signal}_SET"

    _inhibit_suspend__restore_cosmic_idle

    if [[ "${!set_var}" == "1" ]]; then
        _inhibit_suspend__set_status "${status}"
        eval "${!body_var}"
        return $?
    fi

    builtin trap - "${signal}"
    kill -s "${signal}" "$$"
    case "${signal}" in
        INT) exit 130 ;;
        TERM) exit 143 ;;
    esac
}

# Keep the restore hook in front of later caller trap assignments. Bash has no
# trap stack, so this sourced helper shadows `trap` while active and treats
# caller EXIT/INT/TERM handlers as downstream handlers.
trap() {
    if [[ -z "${_INHIBIT_SUSPEND_COSMIC_TOKEN:-}" ]]; then
        # shellcheck disable=SC2064
        builtin trap "$@"
        return
    fi

    if (($# == 0)); then
        builtin trap
        return
    fi

    case "$1" in
        -l | -p)
            # shellcheck disable=SC2064
            builtin trap "$@"
            return
            ;;
    esac

    if [[ "$1" == "--" ]]; then
        shift
    fi
    if (($# == 0)); then
        builtin trap --
        return
    fi

    local action="$1"
    shift || true
    if (($# == 0)); then
        builtin trap -- "${action}"
        return
    fi

    local -a unmanaged_signals=()
    local signal managed_signal
    for signal in "$@"; do
        managed_signal="$(_inhibit_suspend__normalize_managed_signal "${signal}")"
        if [[ -n "${managed_signal}" ]]; then
            _inhibit_suspend__set_previous_trap "${managed_signal}" "${action}"
            _inhibit_suspend__install_managed_trap "${managed_signal}"
        else
            unmanaged_signals+=("${signal}")
        fi
    done

    if ((${#unmanaged_signals[@]} > 0)); then
        builtin trap -- "${action}" "${unmanaged_signals[@]}"
    fi
}
