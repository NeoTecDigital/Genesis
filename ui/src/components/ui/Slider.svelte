<script>
  export let value = 0.5;
  export let min = 0;
  export let max = 1;
  export let step = 0.01;
  
  import { createEventDispatcher } from 'svelte';
  const dispatch = createEventDispatcher();
  
  function handleInput(event) {
    value = parseFloat(event.target.value);
    dispatch('change', { value });
  }
</script>

<input
  type="range"
  bind:value
  {min}
  {max}
  {step}
  on:input={handleInput}
  class="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-genesis-cyan"
/>

