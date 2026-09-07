#!/bin/bash
set -o errexit -o nounset -o pipefail

function main {
	parse_args "$@"
	if [[ $docker -eq 1 ]] ; then
		run_octave
	else
		setup_container
	fi
}

# setup Octave container for specified Matlab script
function setup_container {
	docker build -t octave .
	docker run -it -v `greadlink -m "$script"`:/"$script":ro -e SCRIPT="$script" octave
}

# run Octave within the Docker container
function run_octave {
	octave "$script" --traditional
}

# print program usage
function usage {
	cat << EndOfUsage
Run a Matlab script in a Docker container via Octave
Usage:
  octave_runner [-h] [-d] SCRIPT
Required arguments:
  SCRIPT         path to Matlab script to run
Optional arguments:
  -h, --help     show this menu and exit
  -d, --docker   spin up an Octave container and execute SCRIPT in it
EndOfUsage
}

# parse command line arguments
function parse_args {
	if [[ $# -eq 0 ]] ; then
		usage
		exit 1
	fi

	script=
	docker=0

	while [[ $# -gt 0 ]] ; do
		key="$1"
		case "$key" in
			-d|--docker ) docker=1 ; shift ;;
			-h|--help   ) usage ; exit 0 ;;
			* )
				if [[ "$key" == "-"* ]] ; then
					>&2 echo "-err: unrecognized option '$key'"
					exit 1
				fi
				script="$1" ; shift
				;;
		esac
	done

	if [[ "$script" == "" ]] ; then
		>&2 echo "-err: no script provided"
		exit 1
	fi

	if [[ ! -f "$script" ]] ; then
		>&2 echo "-err: no such file '$script'"
		exit 1
	fi
}

# ======================================= #
if declare -F -- "${1:-}" >/dev/null ; then
	"$@"
else
	main "$@"
fi