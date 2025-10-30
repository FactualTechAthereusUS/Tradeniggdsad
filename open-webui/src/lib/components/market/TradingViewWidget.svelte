<script lang="ts">
  import { onMount } from 'svelte';

  export let symbol: string = 'BINANCE:BTCUSDT';
  export let interval: string = '15';

  let container: HTMLDivElement;

  onMount(() => {
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.type = 'text/javascript';
    script.async = true;

    const config = {
      autosize: true,
      symbol,
      interval,
      timezone: 'Etc/UTC',
      theme: document.documentElement.classList.contains('dark') ? 'dark' : 'light',
      style: '1',
      locale: 'en',
      enable_publishing: false,
      hide_side_toolbar: false,
      allow_symbol_change: true,
      support_host: 'https://www.tradingview.com'
    } as const;

    // Reset and inject
    container.innerHTML = '';
    script.innerHTML = JSON.stringify(config);
    container.appendChild(script);
  });
</script>

<div class="tradingview-widget-container w-full h-full">
  <div bind:this={container} class="tradingview-widget-container__widget w-full h-full"></div>
</div>

<style>
  .tradingview-widget-container,
  .tradingview-widget-container__widget {
    height: 100%;
    width: 100%;
  }
  :global(.tradingview-widget-container iframe) {
    border: 0;
  }
</style>




