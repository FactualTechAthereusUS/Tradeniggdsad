<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { WEBUI_BASE_URL } from '$lib/constants';

    export let symbol: string = 'BTCUSDT';
    export let interval: string = '15m'; // Binance format: 1m, 5m, 15m, 1h, etc.
    let provider: 'binance' | 'yahoo' | 'yfinance' = 'yfinance';

	let container: HTMLDivElement;
    let chart: any = null;
    let series: any = null;
	let ws: WebSocket | null = null;

    // UI State
    let showSearch = false;
    let query = '';
    let cls: 'all'|'crypto'|'stocks'|'forex'|'futures' = 'all';
    let results: Array<{symbol:string, description:string, source:'yahoo'|'binance'}> = [];
    let highlightIdx = 0;
    let loadingSearch = false;
    let debounce: any = null;
    let quickList = [
        'AAPL','MSFT','TSLA','SPY','QQQ',
        'BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT'
    ];
    const timeframes = ['1m','5m','15m','30m','1h','4h','1d'];
    let page = 1;
    let hasMore = true;

	function intervalToMs(iv: string): number {
		const m = iv.match(/^(\d+)([mhd])$/i);
		if (!m) return 60_000;
		const n = parseInt(m[1]);
		switch (m[2].toLowerCase()) {
			case 'm':
				return n * 60_000;
			case 'h':
				return n * 60 * 60_000;
			case 'd':
				return n * 24 * 60 * 60_000;
			default:
				return 60_000;
		}
	}

    function toCandles(klines: any[]): any[] {
		return klines.map((k) => ({
			time: k[0] / 1000,
			open: parseFloat(k[1]),
			high: parseFloat(k[2]),
			low: parseFloat(k[3]),
			close: parseFloat(k[4])
		}));
	}

function normalizeYahooSymbol(s: string): string {
    const u = s.toUpperCase();
    // Map common crypto formats to Yahoo (e.g., BTCUSDT -> BTC-USD)
    if (u.endsWith('USDT')) {
        return `${u.replace('USDT','')}-USD`;
    }
    if (u.endsWith('USD')) {
        return `${u.replace('USD','')}-USD`;
    }
    return u;
}

function isCryptoSymbol(s: string): boolean {
    return /USDT$|USD[TB]$|BTC$|ETH$/i.test(s);
}

function selectProvider(sym: string, iv: string): 'binance'|'yahoo'|'yfinance' {
    const crypto = isCryptoSymbol(sym);
    // Crypto: use Binance for intraday; Yahoo/yfinance for higher TF to get long history
    if (crypto) {
        return iv === '1d' ? 'yfinance' : 'binance';
    }
    // Non-crypto: yfinance
    return 'yfinance';
}

	onMount(async () => {
        // Load UMD if not present (avoids npm dependency)
        if (!(window as any).LightweightCharts) {
            await new Promise<void>((resolve, reject) => {
                const s = document.createElement('script');
                s.src = `${WEBUI_BASE_URL}/static/vendor/lightweight-charts.standalone.production.js`;
                s.async = true;
                s.onload = () => resolve();
                s.onerror = () => reject(new Error('Failed to load lightweight-charts'));
                document.head.appendChild(s);
            });
        }

        const { createChart } = (window as any).LightweightCharts;
        const dark = document.documentElement.classList.contains('dark');
        chart = createChart(container, {
			layout: {
				background: { color: dark ? '#0b0b0b' : '#ffffff' },
				textColor: dark ? '#d1d5db' : '#111827'
			},
			grid: {
				vertLines: { color: dark ? '#1f2937' : '#e5e7eb' },
				horzLines: { color: dark ? '#1f2937' : '#e5e7eb' }
			},
			timeScale: { rightBarStaysOnScroll: true, borderColor: dark ? '#374151' : '#d1d5db' },
			crosshair: { mode: 1 }
		});
        series = chart.addCandlestickSeries({ upColor: '#16a34a', downColor: '#ef4444', wickUpColor: '#16a34a', wickDownColor: '#ef4444', borderVisible: false });

        // Ensure non-zero initial size
        const w = container.clientWidth || 800;
        const h = container.clientHeight || 500;
        chart.resize(w, h);

		// Initial data
        await reloadData();

		// Realtime updates
        subscribeWs();
        (window as any).TradeBergHist = { symbol, interval, provider };

		// Resize
		const ro = new ResizeObserver(() => {
			if (chart && container) chart.resize(container.clientWidth, container.clientHeight);
		});
		ro.observe(container);

		onDestroy(() => {
			ro.disconnect();
			if (ws) try { ws.close(); } catch {}
			chart?.remove();
			chart = null;
		});
	});

    async function reloadData() {
        if (!series) return;
        try {
            const sym = provider === 'yfinance' ? normalizeYahooSymbol(symbol) : symbol;
            const lim = interval === '1d' ? 10000 : 2000;
            const url = `${WEBUI_BASE_URL}/api/markets/candles?provider=${provider}&symbol=${encodeURIComponent(sym)}&interval=${interval}&limit=${lim}`;
            const res = await fetch(url);
            const data = await res.json();
            const arr = data?.candles ?? [];
            const mapped = arr.map((k) => ({ time: k[0]/1000, open: parseFloat(k[1]), high: parseFloat(k[2]), low: parseFloat(k[3]), close: parseFloat(k[4]) }));
            if (mapped.length === 0 && /USDT$|USD[TB]$|BTC$|ETH$/i.test(symbol)) {
                // Fallback to Binance if Yahoo/yfinance has no data for this crypto symbol
                provider = 'binance';
                await reloadData();
                return;
            }
            if (series) series.setData(mapped);
        } catch (e) {
            console.error('REST reload failed', e);
        }
    }

    function subscribeWs() {
        try {
            if (ws) { try { ws.close(); } catch {} ws = null; }
            if (provider === 'binance') {
                const stream = `wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@kline_${interval}`;
                ws = new WebSocket(stream);
                ws.onmessage = (ev) => {
                    const msg = JSON.parse(ev.data);
                    const k = msg.k;
                    if (!k || !series) return;
                    series.update({
                        time: Math.floor(k.t / 1000),
                        open: parseFloat(k.o),
                        high: parseFloat(k.h),
                        low: parseFloat(k.l),
                        close: parseFloat(k.c)
                    });
                };
            } else {
                // yfinance/yahoo: periodic refresh (no WS)
                const timer = setInterval(reloadData, 60_000);
                (chart as any).__timer = timer;
            }
        } catch (e) {
            console.error('WS subscribe failed', e);
        }
    }

    async function setSymbol(next: string) {
        const s = next.toUpperCase();
        symbol = s;
        provider = selectProvider(symbol, interval);
        await reloadData();
        subscribeWs();
        (window as any).TradeBergHist = { symbol, interval, provider };
    }

    async function setIntervalTf(next: string) {
        interval = next;
        provider = selectProvider(symbol, interval);
        await reloadData();
        subscribeWs();
        (window as any).TradeBergHist = { symbol, interval, provider };
    }

    async function runSearch(reset=true) {
        if (reset) { results=[]; page=1; hasMore=true; }
        if (!hasMore || loadingSearch) return;
        loadingSearch = true;
        try {
            const url = `${WEBUI_BASE_URL}/api/markets/search?q=${encodeURIComponent(query)}&cls=${cls}&page=${page}&limit=50`;
            const r = await fetch(url);
            const j = await r.json();
            const arr = j?.results ?? [];
            results = [...results, ...arr];
            if (arr.length < 50) hasMore = false; else page += 1;
            highlightIdx = 0;
        } catch (e) {
            hasMore = false;
        }
        loadingSearch = false;
    }

    function onResultsScroll(e: Event) {
        const el = e.currentTarget as HTMLElement;
        if (el.scrollTop + el.clientHeight >= el.scrollHeight - 20) {
            runSearch(false);
        }
    }

    function onQueryInput() {
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(()=>runSearch(true), 200);
    }

    function openSearch() {
        showSearch = true;
        // load first page for current tab
        results = [];
        page = 1;
        hasMore = true;
        runSearch(true);
    }
</script>

<div class="relative w-full h-full">
    <!-- Toolbar -->
    <div class="absolute top-2 left-2 z-10 flex items-center gap-3 bg-black/40 text-white rounded-xl px-2 py-1 backdrop-blur-md">
        <button class="flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-white/10" on:click={openSearch}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-4 h-4"><path d="M10.5 3a7.5 7.5 0 1 1 0 15 7.5 7.5 0 0 1 0-15Zm8.61 13.5 3.2 3.2a1 1 0 1 1-1.42 1.42l-3.2-3.2a9.5 9.5 0 1 1 1.42-1.42Z"/></svg>
            <span class="font-semibold">{symbol}</span>
        </button>
        <div class="flex items-center gap-1">
            {#each timeframes as tf}
                <button class="px-2 py-0.5 rounded-md text-xs hover:bg-white/10 {interval===tf ? 'bg-white/20' : ''}" on:click={() => setIntervalTf(tf)}>{tf}</button>
            {/each}
        </div>
    </div>

    <!-- Chart -->
    <div id="historical-chart" bind:this={container} class="w-full h-full"></div>

    <!-- Symbol Search Modal -->
    {#if showSearch}
        <div class="absolute inset-0 bg-black/50 backdrop-blur-sm z-20 flex items-start justify-center pt-16" on:click={() => (showSearch=false)}>
            <div class="w-[min(920px,96vw)] bg-[#111315] text-gray-100 rounded-2xl shadow-2xl border border-gray-800" on:click|stopPropagation>
                <div class="px-4 pt-3 pb-2 border-b border-gray-800">
                    <div class="flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-4 h-4 text-gray-400"><path d="M10.5 3a7.5 7.5 0 1 1 0 15 7.5 7.5 0 0 1 0-15Zm8.61 13.5 3.2 3.2a1 1 0 1 1-1.42 1.42l-3.2-3.2a9.5 9.5 0 1 1 1.42-1.42Z"/></svg>
                        <input class="flex-1 bg-transparent outline-none" placeholder="Symbol (e.g. BTCUSDT or AAPL)" bind:value={query} on:input={onQueryInput} on:keydown={(e)=>{
                            if (e.key==='ArrowDown'){ highlightIdx=Math.min(highlightIdx+1, (results?.length||1)-1); e.preventDefault(); }
                            if (e.key==='ArrowUp'){ highlightIdx=Math.max(highlightIdx-1,0); e.preventDefault(); }
                            if (e.key==='Enter'){
                                const r = results[highlightIdx];
                                if(r){ setSymbol(r.symbol); showSearch=false; }
                                else if(query){ setSymbol(query); showSearch=false; }
                            }
                            if (e.key==='Escape'){ showSearch=false; }
                        }}/>
                        <button class="text-sm px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20" on:click={() => (showSearch=false)}>Esc</button>
                    </div>
                    <div class="mt-2 flex items-center gap-2 text-xs">
                        {#each ['all','crypto','stocks','forex','futures'] as tab}
                            <button class="px-2.5 py-1 rounded-full border {cls===tab?'bg-white/10 border-white/20':'border-white/10 hover:bg-white/5'}" on:click={()=>{cls = tab; runSearch(true);}}>{tab.charAt(0).toUpperCase()+tab.slice(1)}</button>
                        {/each}
                    </div>
                </div>
                <div class="h-[70vh] overflow-y-scroll pr-2 tb-scroll" on:scroll={onResultsScroll}>
                    {#if loadingSearch}
                        <div class="px-4 py-3 text-sm text-gray-400">Searching...</div>
                    {:else if results.length>0}
                        {#each results as r, idx}
                            <button class="w-full text-left px-4 py-2 hover:bg-white/5 flex items-center justify-between {idx===highlightIdx?'bg-white/5':''}" on:click={() => { setSymbol(r.symbol); showSearch=false; }}>
                                <div>
                                    <div class="font-medium">{r.symbol}</div>
                                    <div class="text-xs text-gray-400">{r.description}</div>
                                </div>
                                <span class="text-[10px] px-2 py-0.5 rounded-full border border-white/15 text-gray-300">{r.source}</span>
                            </button>
                        {/each}
                    {:else}
                        <div class="px-4 py-3 text-sm text-gray-400">Type to search symbols</div>
                        <div class="px-2 pb-3 grid grid-cols-2 md:grid-cols-3 gap-1">
                            {#each quickList as s}
                                <button class="text-left px-3 py-2 rounded-lg hover:bg-white/5" on:click={() => { setSymbol(s); showSearch=false; }}>
                                    <div class="font-medium">{s}</div>
                                    <div class="text-[11px] text-gray-400">{isCryptoSymbol(s) ? 'spot crypto · Binance' : 'Yahoo'}</div>
                                </button>
                            {/each}
                        </div>
                    {/if}
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
	:global(.tv-hide) { display: none; }
    /* visible thin scrollbar for symbol panel */
    .tb-scroll { scrollbar-width: thin; scrollbar-color: #6b7280 #111315; }
    .tb-scroll::-webkit-scrollbar { width: 8px; }
    .tb-scroll::-webkit-scrollbar-track { background: #111315; border-radius: 8px; }
    .tb-scroll::-webkit-scrollbar-thumb { background: #6b7280; border-radius: 8px; }
</style>
