#!/bin/bash

start=2018-11-22T15:00:00
reservation=tau
partitions="knm big p"

get_valid_users() {
    v="tau"
    for u in $(grep -v '#' users.txt); do
        if id ${u} > /dev/null; then
            v="${v},${u}"
        fi
    done
    echo ${v}
}

create() {
    users=$(get_valid_users)
    for p in ${partitions} ; do
        sudo scontrol create reservation reservation=${p}${reservation} user=${users} starttime=${start} duration=300 nodecnt=1 partition=${p}
    done
}

delete() {
    for p in ${partitions} ; do
        sudo scontrol delete reservation=${p}${reservation}
    done
}

update() {
    users=$(get_valid_users)
    for p in ${partitions} ; do
        sudo scontrol update reservation=${p}${reservation} user=${users}
    done
}

show() {
    sudo scontrol show reservation
}

case "$1" in
    c*)
        create ;;
    d*)
        delete ;;
    u*)
        update ;;
    g*)
        get_valid_users ;;
    *)
        show ;;
esac
