#!/bin/bash

# Standard MOOS-IvP vars
TIME_WARP=1
SHORE_IP=localhost
SHORE_LISTEN="9300"
SHARE_LISTEN=""
VPORT=""
VNAME=""

HELP="no"
JUST_BUILD="no"

# Aquaticus specific
START_POS=""
VTEAM=""
OWN_FLAG=""
ENEMY_FLAG=""
GRABR_POS=""
GRABL_POS=""
LOITER1=""
LOITER2=""

BEHAVIOR="DEFEND"
RED_FLAG="50,-24"
BLUE_FLAG="-52,-70"

RED_LOITER1="30,-60"
RED_LOITER2="14,-22"

BLUE_LOITER1="-28,-42"
BLUE_LOITER2="-12,-79"

# MOOS-IvP-Agent specific
ROLE=""
ID=""
COLOR="green"
LOGGING="no"
NO_AGENT="false"
SCENARIO="attack"

function help(){
    echo ""
    echo "USAGE: $0 <team> <role> <id> [SWITCHES]"

    echo ""
    echo "POSSIBLE TEAMS:"
    echo "  red,          r  : Red team."
    echo "  blue,         b  : Blue team."

    echo ""
    echo "POSSIBLE ROLES:"
    echo "  agent,        a  : Vehicle running behavior of interest."
    echo "  drone,        d  : Vehicle running supporting behavior."
    echo "                For example, a behavior to train against."

    echo ""
    echo "POSSIBLE IDs: [11,99]"

    echo ""
    echo "POSSIBLE SWITCHES:"
    echo "  --no_agent            : No BHV_Agent "
    echo "  --just_build, -J      : Just build targ files."
    echo "  --help,       -H      : Display this message."
    echo "  --behavior=<behavior> : Set the vehicle's color"
    echo "  --color=<some_color>  : Set the vehicle's color"
    exit 0
}

#-------------------------------------------------------
#  Part 1: Check for and handle command-line arguments
#-------------------------------------------------------

# Handle teams
case "$1" in
    red|r)
        VTEAM="red"
        echo "Vehicle added to red team."
        ;;
    blue|b)
        VTEAM="blue"
        echo "Vehicle added to blue team."
        ;;
    *)
        echo "!!! ERROR: expected team assignment got: $1 !!!"
        help
        ;;
esac

if [ "${VTEAM}" = "red" ]; then
    MY_FLAG=$RED_FLAG
    START_POS="$RED_FLAG,240"
    ENEMY_FLAG=$BLUE_FLAG

    GRABR_POS="-46,-42"
    GRABL_POS="-29,-83"

    LOITER1=$RED_LOITER1
    LOITER2=$RED_LOITER2
elif [ "${VTEAM}" = "blue" ]; then
    MY_FLAG=$BLUE_FLAG
    START_POS="$BLUE_FLAG,60"
    ENEMY_FLAG=$RED_FLAG

    GRABR_POS="42,-55"
    GRABL_POS="19,-11"

    LOITER1=$BLUE_LOITER1
    LOITER2=$BLUE_LOITER2
fi

# Handle role assigment
case "$2" in
    agent|a)
        ROLE="agent"
        BEHAVIOR="AGENT"
        echo "Vehicle set as an agent."
        ;;
    drone|d)
        ROLE="drone"
        echo "Vehicle set as a drone."
        ;;
    *)
        echo "!!! ERROR: expected role assignment got: $2 !!!"
        help
        ;;
esac

if [[ "$3" =~ ^-?[0-9]+$ ]] && [[ "$3" -ge 11 ]] && [[ "$3" -le 99 ]]; then
    ID="$3"
    SHARE_LISTEN="93${ID}"
    VPORT="93${ID}"
else
    help
fi

# Set VNAME based on role and id
VNAME="${ROLE}_${ID}"

for arg in "${@:4}"; do
    if [ "${arg}" = "--help" -o "${arg}" = "-H" ]; then
        help
    elif [ "${arg//[^0-9]/}" = "$arg" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$arg
        echo "Time warp set to: " $arg
    elif [ "${arg}" = "--just_build" -o "${arg}" = "-J" ] ; then
        JUST_BUILD="yes"
        echo "Just building files; no vehicle launch."
    elif [ "${arg}" = "--no_agent" ]; then
        NO_AGENT="true"
    elif [ "${arg}" = "--log" ] ; then
        LOGGING="yes"
    elif [ "${arg:0:8}" = "--color=" ]; then
        COLOR="${arg#--color=*}"
    elif [ "${arg:0:11}" = "--behavior=" ]; then
        BEHAVIOR="${arg#--behavior=*}"
    elif [ "${arg:0:11}" = "--scenario=" ]; then
        SCENARIO="${arg#--scenario=*}"
    else
        echo "Undefined switch:" $arg
        help
    fi
done

#-------------------------------------------------------
#  Part 2: Create the .moos and .bhv files.
#-------------------------------------------------------

echo "Assembling MOOS file targ_${VNAME}.moos"
nsplug meta_heron.moos targ_${VNAME}.moos -f \
    SHORE_IP=$SHORE_IP           \
    SHORE_LISTEN=$SHORE_LISTEN   \
    SHARE_LISTEN=$SHARE_LISTEN   \
    VPORT=$VPORT                 \
    VNAME=$VNAME                 \
    WARP=$TIME_WARP              \
    VTYPE="kayak"                \
    VTEAM=$VTEAM                 \
    START_POS=$START_POS         \
    COLOR=$COLOR                 \
    LOGGING=$LOGGING             \
    ROLE=$ROLE                   \
    SCENARIO=$SCENARIO           \

echo "Assembling BHV file targ_${VNAME}.bhv"
nsplug meta_heron.bhv targ_${VNAME}.bhv -f  \
        VTEAM=$VTEAM                        \
        VNAME=$VNAME                        \
        ROLE=$ROLE                          \
        MY_FLAG=$MY_FLAG                    \
        ENEMY_FLAG=$ENEMY_FLAG              \
        GRABR_POS=$GRABR_POS                \
        GRABL_POS=$GRABL_POS                \
        LOITER1=$LOITER1                    \
        LOITER2=$LOITER2                    \
        NO_AGENT=$NO_AGENT                  \
	    BEHAVIOR=$BEHAVIOR


if [ ${JUST_BUILD} = "yes" ] ; then
    echo "Files assembled; vehicle not launched; exiting per request."
    exit 0
fi

#-------------------------------------------------------
#  Part 3: Launch the processes
#-------------------------------------------------------

echo "Launching $VNAME MOOS Community "
pAntler targ_${VNAME}.moos >& /dev/null &

uMAC targ_${VNAME}.moos

echo "Killing all processes ..."
kill -- -$$
echo "Done killing processes."
