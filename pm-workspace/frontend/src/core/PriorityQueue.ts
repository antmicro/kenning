/*
 * Copyright (c) 2026 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Priority Queue implementation for usage in A* algorithm
 * */

type PriorityQueueItem<T> = {
    value: T;
    priority: number;
}

export default class PriorityQueue<T> {
    private heap: PriorityQueueItem<T>[] = [];

    private indices = new Map<T, number>();

    enqueue(value: T, priority: number): void {
        if (this.indices.has(value)) return;

        this.heap.push({ value, priority });
        const index = this.heap.length - 1;
        this.indices.set(value, index);
        this.bubbleUp(index);
    }

    contains(value: T): boolean {
        return this.indices.has(value);
    }

    updatePrio(value: T, newPriority: number): boolean {
        const index = this.indices.get(value);

        if (index === undefined) return false;

        const oldPriority = this.heap[index].priority;
        this.heap[index].priority = newPriority;

        if (newPriority < oldPriority) {
            this.bubbleUp(index);
        } else {
            this.bubbleDown(index);
        }
        return true;
    }

    dequeue(): T | null {
        if (this.heap.length === 0) return null;

        const min = this.heap[0];
        const end = this.heap.pop()!;

        this.indices.delete(min.value);

        if (this.heap.length > 0) {
            this.heap[0] = end;
            this.indices.set(end.value, 0);
            this.bubbleDown(0);
        }

        return min.value;
    }

    peek(): T | null {
        return this.heap.length ? this.heap[0].value : null;
    }

    size(): number {
        return this.heap.length;
    }

    isEmpty(): boolean {
        return this.heap.length === 0;
    }

    private swap(i: number, j: number): void {
        [this.heap[i], this.heap[j]] = [
            this.heap[j],
            this.heap[i],
        ];

        this.indices.set(this.heap[i].value, i);
        this.indices.set(this.heap[j].value, j);
    }

    private bubbleUp(ind: number): void {
        let index = ind;
        while (index > 0) {
            const parentIndex = Math.floor((index - 1) / 2);

            if (this.heap[parentIndex].priority <= this.heap[index].priority) {
                break;
            }

            this.swap(index, parentIndex);

            index = parentIndex;
        }
    }

    private bubbleDown(ind: number): void {
        let index = ind;
        const { length } = this.heap;

        while (true) {
            const left = index * 2 + 1;
            const right = index * 2 + 2;
            let smallest = index;

            if (
                left < length &&
                this.heap[left].priority <
                  this.heap[smallest].priority
            ) {
                smallest = left;
            }

            if (
                right < length &&
                this.heap[right].priority <
                  this.heap[smallest].priority
            ) {
                smallest = right;
            }

            if (smallest === index) break;

            this.swap(index, smallest);

            index = smallest;
        }
    }
}
