import { writable } from 'svelte/store';

export const filterState = writable({
  gamma: true,
  iota: true,
  tau: true,
  epsilon: true,
  protoIdentity: true,
  instances: true
});

