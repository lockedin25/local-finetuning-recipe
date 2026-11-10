#!/bin/zsh
# Live dashboard for baseline / training runs. Ctrl+C to quit (does NOT stop the run).
cd "${0:A:h}/.."
LOG=${1:-module4_5.log}
while true; do
  clear
  echo "=== $(date +%H:%M:%S)   watching $LOG ==="
  echo
  pid=$(pgrep -f "python3? src/(bakeoff|train).py" | head -1)
  if [[ -n $pid ]]; then
    ps -o etime=,%cpu=,rss=,command= -p $pid | awk '{printf "RUNNING  %s   elapsed %s   cpu %s%%   memory %.1f GB\n", $5, $1, $2, $3/1048576}'
  else
    echo "no bakeoff/train process running"
  fi
  memory_pressure | tail -1 | sed 's/System-wide memory free percentage/Mac memory free/'
  sysctl -n vm.swapusage | awk '{print "swap used: " $6}'
  echo
  echo "--- last lines of the log ---"
  tail -15 "$LOG" 2>/dev/null
  sleep 3
done
