import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage.ts"
import { TradingEngine } from "./engine.ts"
//import { backtestEngine } from "./backtest.ts";
import ccxt from "ccxt";
import OpenAI from "openai";
//import { setupAuth, registerAuthRoutes, isAuthenticated } from "./replit_integrations/auth";
import { getGitHubUser, createRepo, getRepo, pushFile, getFileContent } from "./github";
import * as fs from "fs";
import * as path from "path";

const openai = new OpenAI({
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Setup Replit Auth (must be before other routes)
  await setupAuth(app);
  registerAuthRoutes(app);

  const engine = TradingEngine.getInstance();

  // Helper to get or verify user's bot
  async function getUserBot(req: any, res: any): Promise<any | null> {
    const userId = req.user?.claims?.sub;
    if (!userId) {
      res.status(401).json({ message: "Not authenticated" });
      return null;
    }
    const bot = await storage.getOrCreateBotForUser(userId);
    return bot;
  }

  // Helper to verify bot ownership
  async function verifyBotOwnership(req: any, res: any, botId: number): Promise<any | null> {
    const userId = req.user?.claims?.sub;
    if (!userId) {
      res.status(401).json({ message: "Not authenticated" });
      return null;
    }
    const bot = await storage.getBot(botId);
    if (!bot) {
      res.status(404).json({ message: "Bot not found" });
      return null;
    }
    if (bot.userId !== userId) {
      res.status(403).json({ message: "Access denied" });
      return null;
    }
    return bot;
  }

  // Get user's bot (creates one if doesn't exist)
  app.get("/api/my-bot", isAuthenticated, async (req: any, res) => {
    const bot = await getUserBot(req, res);
    if (!bot) return;
    res.json(bot);
  });

  app.get("/api/bot/:id", isAuthenticated, async (req: any, res) => {
   // const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    res.json(bot);
  });

  app.get("/api/bot/:id/backtest", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const backtests = await storage.getBacktests(bot.id);
    res.json(backtests);
  });

  app.post("/api/bot/:id/backtest", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const days = req.body.days || 7;
    try {
      const result = await backtestEngine.runBacktest(bot.id, days);
      const saved = await storage.createBacktest(result);
      res.json(saved);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  app.patch("/api/bot/:id", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const updated = await storage.updateBot(bot.id, req.body);
    
    if (updated.isRunning) {
      engine.start(bot.id);
    } else {
      engine.stop(bot.id);
    }
    
    res.json(updated);
  });

  app.get("/api/bot/:id/trades", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const trades = await storage.getTrades(bot.id);
    res.json(trades);
  });

  app.get("/api/bot/:id/logs", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const logs = await storage.getLogs(bot.id);
    res.json(logs);
  });

  app.post("/api/bot/:id/kill", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    await storage.updateBot(bot.id, { isRunning: false, isLiveMode: false });
    engine.stop(bot.id);
    res.json({ message: "Emergency shutdown complete" });
  });

  // Paper trading reset and configuration
  app.post("/api/bot/:id/paper/reset", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const { startingCapital } = req.body;
    const capital = startingCapital || 10000;
    
    // Delete only paper trades (preserve live trades)
    await storage.clearPaperTrades(bot.id);
    
    // Reset only paper trading stats (preserve live trading metrics)
    await storage.updateBot(bot.id, {
      paperStartingCapital: capital,
      paperBalance: capital,
      paperTotalFees: 0,
      paperTotalSlippage: 0,
      paperWinCount: 0,
      paperLossCount: 0,
      paperBestTrade: 0,
      paperWorstTrade: 0,
      paperStartedAt: new Date(),
    });
    
    await storage.createLog({
      botId: bot.id,
      level: 'info',
      message: `Paper trading reset with $${capital.toLocaleString()} starting capital`
    });
    
    res.json({ 
      message: "Paper trading reset",
      startingCapital: capital 
    });
  });

  // Fetch live account balance from exchange
  app.get("/api/bot/:id/account/balance", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const botId = bot.id;

    try {
      // If in paper mode, return paper balance
      if (!bot.isLiveMode) {
        return res.json({
          isLive: false,
          balance: bot.paperBalance || 10000,
          startingCapital: bot.paperStartingCapital || 10000,
          currency: 'USD',
          holdings: [],
          totalEquity: bot.paperBalance || 10000,
          source: 'paper'
        });
      }

      // In live mode, fetch from exchange
      const exchangeName = bot.exchange || 'coinbase';
      let exchange: any;
      
      if (exchangeName === 'coinbase') {
        if (!bot.coinbaseApiKey || !bot.coinbaseApiSecret) {
          return res.status(400).json({ 
            message: "Coinbase API keys not configured. Please add your API keys in Settings.",
            needsKeys: true
          });
        }
        // Handle escaped newlines in the secret key
        const secret = bot.coinbaseApiSecret.replace(/\\n/g, '\n');
        exchange = new ccxt.coinbase({
          apiKey: bot.coinbaseApiKey,
          secret: secret,
          enableRateLimit: true,
        });
      } else if (exchangeName === 'kraken') {
        if (!bot.krakenApiKey || !bot.krakenApiSecret) {
          return res.status(400).json({ 
            message: "Kraken API keys not configured. Please add your API keys in Settings.",
            needsKeys: true
          });
        }
        // Handle escaped newlines in the secret key
        const secret = bot.krakenApiSecret.replace(/\\n/g, '\n');
        exchange = new ccxt.kraken({
          apiKey: bot.krakenApiKey,
          secret: secret,
          enableRateLimit: true,
        });
      }

      const balanceData = await exchange.fetchBalance();
      
      // Calculate total USD value
      let totalUSD = 0;
      const holdings: { currency: string; amount: number; usdValue: number }[] = [];
      
      // Get USD/USDT balance
      const usdBalance = (balanceData.USD?.free || 0) + (balanceData.USDT?.free || 0);
      totalUSD += usdBalance;
      
      if (usdBalance > 0) {
        holdings.push({ currency: 'USD', amount: usdBalance, usdValue: usdBalance });
      }

      // Get crypto holdings and their USD values
      for (const [currency, data] of Object.entries(balanceData.total || {})) {
        const amount = data as number;
        if (amount > 0 && currency !== 'USD' && currency !== 'USDT') {
          try {
            const ticker = await exchange.fetchTicker(`${currency}/USDT`);
            const usdValue = amount * (ticker.last || 0);
            if (usdValue > 1) { // Only include if worth more than $1
              holdings.push({ currency, amount, usdValue });
              totalUSD += usdValue;
            }
          } catch {
            // Skip if can't get price
          }
        }
      }

      // Sort holdings by value
      holdings.sort((a, b) => b.usdValue - a.usdValue);

      // Calculate unrealized P&L from open positions
      let unrealizedPnL = 0;
      const openPositions: { symbol: string; entryPrice: number; quantity: number; currentPrice: number; unrealizedPnl: number }[] = [];
      
      try {
        const trades = await storage.getTrades(botId);
        const openTrades = trades.filter((t: any) => t.side === 'buy' && t.pnl === null);
        
        for (const trade of openTrades) {
          try {
            const ticker = await exchange.fetchTicker(trade.symbol);
            const currentPrice = ticker.last || 0;
            const entryPrice = Number(trade.price) || 0;
            const quantity = Number(trade.amount) || 0;
            const pnl = (currentPrice - entryPrice) * quantity;
            
            unrealizedPnL += pnl;
            openPositions.push({
              symbol: trade.symbol,
              entryPrice,
              quantity,
              currentPrice,
              unrealizedPnl: pnl
            });
          } catch {
            // Skip if can't fetch price for this position
          }
        }
      } catch (err) {
        console.error('Error calculating unrealized P&L:', err);
      }

      // Calculate realized P&L for today
      let todayRealizedPnL = 0;
      try {
        const allTrades = await storage.getTrades(botId);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        const todayTrades = allTrades.filter((t: any) => {
          const tradeDate = new Date(t.timestamp);
          return tradeDate >= today && t.pnl !== null;
        });
        
        todayRealizedPnL = todayTrades.reduce((acc: number, t: any) => acc + (Number(t.pnl) || 0), 0);
      } catch (err) {
        console.error('Error calculating realized P&L:', err);
      }

      // Update live equity history (track balance over time)
      const liveHistory = [...(bot.liveEquityHistory || [])];
      const now = new Date().toISOString();
      
      // Only add new data point if last one is more than 1 minute old
      const lastEntry = liveHistory[liveHistory.length - 1];
      const shouldAddEntry = !lastEntry || 
        (new Date(now).getTime() - new Date(lastEntry.timestamp).getTime() > 60000);
      
      if (shouldAddEntry) {
        liveHistory.push({ timestamp: now, balance: totalUSD });
        // Keep last 100 data points
        if (liveHistory.length > 100) liveHistory.shift();
        
        await storage.updateBot(botId, { liveEquityHistory: liveHistory });
      }

      res.json({
        isLive: true,
        balance: totalUSD,
        startingCapital: bot.paperStartingCapital || 10000, // Use paper starting as reference
        currency: 'USD',
        holdings: holdings.slice(0, 10), // Top 10 holdings
        totalEquity: totalUSD,
        source: exchangeName,
        equityHistory: liveHistory, // Include live equity history
        unrealizedPnL,
        todayRealizedPnL,
        todayTotalPnL: todayRealizedPnL + unrealizedPnL,
        openPositions
      });
    } catch (error: any) {
      console.error('Balance fetch error:', error);
      await storage.createLog({
        botId,
        level: 'error',
        message: `Failed to fetch live balance: ${error.message}`
      });
      res.status(500).json({ 
        message: `Failed to fetch balance: ${error.message}`,
        error: true
      });
    }
  });

  // Check exchange connection status
  app.get("/api/bot/:id/exchange/status", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;

    const exchangeName = bot.exchange || 'coinbase';
    
    try {
      // Check if API keys are configured
      const hasKeys = exchangeName === 'coinbase' 
        ? !!(bot.coinbaseApiKey && bot.coinbaseApiSecret)
        : !!(bot.krakenApiKey && bot.krakenApiSecret);

      // Get connection status from engine if available
      const engineStatus = engine.getConnectionStatus(exchangeName);

      if (!hasKeys) {
        return res.json({
          exchange: exchangeName,
          connected: false,
          status: 'not_configured',
          message: 'API keys not configured',
          isLiveMode: bot.isLiveMode,
          health: {
            configured: false,
            consecutiveFailures: 0,
            lastError: null,
            recommendation: 'Configure API keys in Settings to enable live trading'
          }
        });
      }

      // Test connection by fetching balance
      let exchange: any;
      if (exchangeName === 'coinbase') {
        const secret = bot.coinbaseApiSecret!.replace(/\\n/g, '\n');
        exchange = new ccxt.coinbase({
          apiKey: bot.coinbaseApiKey!,
          secret: secret,
          enableRateLimit: true,
          timeout: 10000,
        });
      } else {
        const secret = bot.krakenApiSecret!.replace(/\\n/g, '\n');
        exchange = new ccxt.kraken({
          apiKey: bot.krakenApiKey!,
          secret: secret,
          enableRateLimit: true,
          timeout: 10000,
        });
      }

      const startTime = Date.now();
      const balance = await exchange.fetchBalance();
      const latency = Date.now() - startTime;

      // Determine health status
      let healthStatus = 'excellent';
      let recommendation = 'Connection is stable and ready for live trading';
      
      if (latency > 5000) {
        healthStatus = 'poor';
        recommendation = 'High latency detected. Consider checking your network or using a different exchange';
      } else if (latency > 2000) {
        healthStatus = 'fair';
        recommendation = 'Moderate latency. Trading should work but may experience delays';
      } else if (latency > 1000) {
        healthStatus = 'good';
        recommendation = 'Connection is acceptable for trading';
      }

      // Check if there have been recent failures
      if (engineStatus && engineStatus.consecutiveFailures > 0) {
        healthStatus = 'recovering';
        recommendation = `Connection recovering from ${engineStatus.consecutiveFailures} recent failures`;
      }

      res.json({
        exchange: exchangeName,
        connected: true,
        status: 'connected',
        message: `Connected to ${exchangeName}`,
        latency: latency,
        isLiveMode: bot.isLiveMode,
        lastChecked: new Date().toISOString(),
        health: {
          status: healthStatus,
          configured: true,
          consecutiveFailures: engineStatus?.consecutiveFailures || 0,
          lastConnected: engineStatus?.lastConnected?.toISOString() || new Date().toISOString(),
          lastError: engineStatus?.lastError || null,
          recommendation: recommendation,
          latencyMs: latency,
          latencyRating: latency < 1000 ? 'fast' : latency < 3000 ? 'moderate' : 'slow'
        },
        balanceAvailable: {
          USD: balance.total?.USD || balance.total?.USDT || 0,
          hasBalance: (balance.total?.USD || balance.total?.USDT || 0) > 0
        }
      });
    } catch (error: any) {
      const errorType = error.constructor?.name || 'UnknownError';
      let recommendation = 'Check your API keys and network connection';
      
      if (error.message?.includes('authentication') || error.message?.includes('API')) {
        recommendation = 'API key authentication failed. Please verify your API credentials in Settings';
      } else if (error.message?.includes('timeout')) {
        recommendation = 'Connection timed out. The exchange may be experiencing issues';
      } else if (error.message?.includes('rate limit')) {
        recommendation = 'Rate limit exceeded. Wait a few minutes before trying again';
      }

      res.json({
        exchange: exchangeName,
        connected: false,
        status: 'error',
        message: error.message,
        isLiveMode: bot.isLiveMode,
        health: {
          status: 'error',
          configured: true,
          errorType: errorType,
          consecutiveFailures: (engine.getConnectionStatus(exchangeName)?.consecutiveFailures || 0) + 1,
          lastError: error.message,
          recommendation: recommendation
        }
      });
    }
  });

  // Fetch order book depth for live trading visualization
  app.get("/api/bot/:id/orderbook", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;

    if (!bot.isLiveMode) {
      return res.json({ message: 'Order book only available in live mode' });
    }

    const exchangeName = bot.exchange || 'coinbase';
    
    // Validate API keys are configured
    const hasKeys = exchangeName === 'coinbase' 
      ? !!(bot.coinbaseApiKey && bot.coinbaseApiSecret)
      : !!(bot.krakenApiKey && bot.krakenApiSecret);

    if (!hasKeys) {
      return res.status(400).json({ 
        message: "API keys not configured", 
        needsKeys: true 
      });
    }

    try {
      let exchange: any;
      
      if (exchangeName === 'coinbase') {
        const secret = bot.coinbaseApiSecret!.replace(/\\n/g, '\n');
        exchange = new ccxt.coinbase({
          apiKey: bot.coinbaseApiKey!,
          secret: secret,
          enableRateLimit: true,
          timeout: 10000,
        });
      } else {
        const secret = bot.krakenApiSecret!.replace(/\\n/g, '\n');
        exchange = new ccxt.kraken({
          apiKey: bot.krakenApiKey!,
          secret: secret,
          enableRateLimit: true,
          timeout: 10000,
        });
      }

      const orderBook = await exchange.fetchOrderBook(bot.symbol, 10);
      
      const bids = orderBook.bids.slice(0, 5);
      const asks = orderBook.asks.slice(0, 5);
      
      const bestBid = bids[0]?.[0] || 0;
      const bestAsk = asks[0]?.[0] || 0;
      const spread = bestAsk - bestBid;
      const midPrice = (bestBid + bestAsk) / 2;
      const spreadPercent = midPrice > 0 ? (spread / midPrice) * 100 : 0;
      
      const bidVolume = bids.reduce((sum: number, b: [number, number]) => sum + b[1], 0);
      const askVolume = asks.reduce((sum: number, a: [number, number]) => sum + a[1], 0);
      const totalVolume = bidVolume + askVolume;
      const imbalance = totalVolume > 0 ? (bidVolume - askVolume) / totalVolume : 0;

      res.json({
        bids,
        asks,
        spread,
        spreadPercent,
        midPrice,
        bidVolume,
        askVolume,
        imbalance,
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      res.status(500).json({ 
        message: "Failed to fetch order book", 
        error: error.message 
      });
    }
  });

  // Fetch recent orders from exchange
  app.get("/api/bot/:id/exchange/orders", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;

    if (!bot.isLiveMode) {
      return res.json({ orders: [], message: 'Paper trading mode - no live orders' });
    }

    const exchangeName = bot.exchange || 'coinbase';
    
    try {
      let exchange: any;
      if (exchangeName === 'coinbase') {
        if (!bot.coinbaseApiKey || !bot.coinbaseApiSecret) {
          return res.status(400).json({ message: "API keys not configured" });
        }
        const secret = bot.coinbaseApiSecret.replace(/\\n/g, '\n');
        exchange = new ccxt.coinbase({
          apiKey: bot.coinbaseApiKey,
          secret: secret,
          enableRateLimit: true,
        });
      } else {
        if (!bot.krakenApiKey || !bot.krakenApiSecret) {
          return res.status(400).json({ message: "API keys not configured" });
        }
        const secret = bot.krakenApiSecret.replace(/\\n/g, '\n');
        exchange = new ccxt.kraken({
          apiKey: bot.krakenApiKey,
          secret: secret,
          enableRateLimit: true,
        });
      }

      // Fetch open orders
      const openOrders = await exchange.fetchOpenOrders(bot.symbol);
      
      // Fetch closed orders (recent)
      let closedOrders: any[] = [];
      try {
        closedOrders = await exchange.fetchClosedOrders(bot.symbol, undefined, 20);
      } catch {
        // Some exchanges don't support fetchClosedOrders
      }

      res.json({
        openOrders: openOrders.map((o: any) => ({
          id: o.id,
          symbol: o.symbol,
          side: o.side,
          type: o.type,
          price: o.price,
          amount: o.amount,
          filled: o.filled,
          status: o.status,
          timestamp: o.timestamp
        })),
        recentOrders: closedOrders.slice(0, 10).map((o: any) => ({
          id: o.id,
          symbol: o.symbol,
          side: o.side,
          type: o.type,
          price: o.price || o.average,
          amount: o.amount,
          filled: o.filled,
          status: o.status,
          timestamp: o.timestamp
        })),
        exchange: exchangeName
      });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Fetch account deposit/withdraw history
  app.get("/api/bot/:id/exchange/transactions", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;

    if (!bot.isLiveMode) {
      return res.json({ transactions: [], message: 'Paper trading mode' });
    }

    const exchangeName = bot.exchange || 'coinbase';
    
    try {
      let exchange: any;
      if (exchangeName === 'coinbase') {
        if (!bot.coinbaseApiKey || !bot.coinbaseApiSecret) {
          return res.status(400).json({ message: "API keys not configured" });
        }
        const secret = bot.coinbaseApiSecret.replace(/\\n/g, '\n');
        exchange = new ccxt.coinbase({
          apiKey: bot.coinbaseApiKey,
          secret: secret,
          enableRateLimit: true,
        });
      } else {
        if (!bot.krakenApiKey || !bot.krakenApiSecret) {
          return res.status(400).json({ message: "API keys not configured" });
        }
        const secret = bot.krakenApiSecret.replace(/\\n/g, '\n');
        exchange = new ccxt.kraken({
          apiKey: bot.krakenApiKey,
          secret: secret,
          enableRateLimit: true,
        });
      }

      // Fetch deposits and withdrawals
      let deposits: any[] = [];
      let withdrawals: any[] = [];
      
      try {
        deposits = await exchange.fetchDeposits();
        withdrawals = await exchange.fetchWithdrawals();
      } catch {
        // Not all exchanges support this
      }

      res.json({
        deposits: deposits.slice(0, 10).map((d: any) => ({
          id: d.id,
          currency: d.currency,
          amount: d.amount,
          status: d.status,
          timestamp: d.timestamp
        })),
        withdrawals: withdrawals.slice(0, 10).map((w: any) => ({
          id: w.id,
          currency: w.currency,
          amount: w.amount,
          status: w.status,
          timestamp: w.timestamp
        })),
        exchange: exchangeName
      });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.patch("/api/bot/:id/paper/settings", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    const { simulatedFeeRate, simulatedSlippageRate } = req.body;
    
    const updates: Record<string, number> = {};
    if (simulatedFeeRate !== undefined) {
      updates.simulatedFeeRate = Math.max(0, Math.min(0.01, simulatedFeeRate)); // 0-1%
    }
    if (simulatedSlippageRate !== undefined) {
      updates.simulatedSlippageRate = Math.max(0, Math.min(0.005, simulatedSlippageRate)); // 0-0.5%
    }
    
    if (Object.keys(updates).length > 0) {
      await storage.updateBot(bot.id, updates);
    }
    
    const updatedBot = await storage.getBot(bot.id);
    res.json({ 
      simulatedFeeRate: updatedBot?.simulatedFeeRate,
      simulatedSlippageRate: updatedBot?.simulatedSlippageRate
    });
  });

  app.get("/api/bot/:id/paper/stats", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    const trades = await storage.getTrades(bot.id);
    const paperTrades = trades.filter(t => t.isPaperTrade);
    
    const winRate = bot.paperWinCount && (bot.paperWinCount + (bot.paperLossCount || 0)) > 0
      ? (bot.paperWinCount / (bot.paperWinCount + (bot.paperLossCount || 0))) * 100
      : 0;
    
    const profitFactor = paperTrades.length > 0 
      ? (() => {
          const grossProfit = paperTrades.filter(t => (t.pnl || 0) > 0).reduce((sum, t) => sum + (t.pnl || 0), 0);
          const grossLoss = Math.abs(paperTrades.filter(t => (t.pnl || 0) < 0).reduce((sum, t) => sum + (t.pnl || 0), 0));
          return grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
        })()
      : 0;
    
    res.json({
      startingCapital: bot.paperStartingCapital || 10000,
      currentBalance: bot.paperBalance || 10000,
      totalPnL: (bot.paperBalance || 10000) - (bot.paperStartingCapital || 10000),
      totalReturn: ((bot.paperBalance || 10000) - (bot.paperStartingCapital || 10000)) / (bot.paperStartingCapital || 10000) * 100,
      totalFees: bot.paperTotalFees || 0,
      totalSlippage: bot.paperTotalSlippage || 0,
      winCount: bot.paperWinCount || 0,
      lossCount: bot.paperLossCount || 0,
      winRate,
      profitFactor,
      bestTrade: bot.paperBestTrade || 0,
      worstTrade: bot.paperWorstTrade || 0,
      totalTrades: (bot.paperWinCount || 0) + (bot.paperLossCount || 0),
      simulatedFeeRate: bot.simulatedFeeRate || 0.001,
      simulatedSlippageRate: bot.simulatedSlippageRate || 0.0005,
      startedAt: bot.paperStartedAt
    });
  });

  // GPT Chat endpoint
  app.post("/api/bot/:id/chat", isAuthenticated, async (req: any, res) => {
    try {
      const verifiedBot = await verifyBotOwnership(req, res, parseInt(req.params.id));
      if (!verifiedBot) return;
      const { message, history } = req.body;
      
      const bot = verifiedBot;
      const trades = await storage.getTrades(bot.id);
      
      const systemPrompt = `You are an expert AI trading assistant for a crypto trading bot named "${bot?.name || 'Astraeus AI'}". 
You help users understand trading strategies, market analysis, and their bot's performance.

Current bot status:
- Symbol: ${bot?.symbol || 'BTC/USDT'}
- Running: ${bot?.isRunning ? 'Yes' : 'No'}
- Mode: ${bot?.isLiveMode ? 'LIVE' : 'Paper Trading'}
- Risk Profile: ${bot?.riskProfile || 'balanced'}
- Current RSI: ${bot?.currentRsi?.toFixed(1) || 'N/A'}
- AI Confidence: ${((bot?.aiConfidence || 0) * 100).toFixed(0)}%
- Total Trades: ${trades.length}
- Win Rate: ${trades.filter(t => (t.pnl || 0) > 0).length}/${trades.length}

Provide helpful, educational responses about crypto trading. Be concise but informative.`;

      const messages = [
        { role: "system" as const, content: systemPrompt },
        ...(history || []).map((m: {role: string, content: string}) => ({
          role: m.role as "user" | "assistant",
          content: m.content
        })),
        { role: "user" as const, content: message }
      ];

      const completion = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages,
        max_completion_tokens: 500
      });

      res.json({
        response: completion.choices[0].message.content || "I couldn't generate a response."
      });
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  app.get("/api/market/:symbol/ohlcv", isAuthenticated, async (req, res) => {
    try {
      const symbol = decodeURIComponent(req.params.symbol);
      const exchange = new ccxt.coinbase({ enableRateLimit: true, timeout: 15000 });
      const ohlcv = await exchange.fetchOHLCV(symbol, '1h', undefined, 100);
      const data = ohlcv.map(c => ({
        time: Math.floor((c[0] || 0) / 1000),
        open: c[1] || 0,
        high: c[2] || 0,
        low: c[3] || 0,
        close: c[4] || 0,
        volume: c[5] || 0
      }));
      res.json(data);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  app.get("/api/market/:symbol/rsi", isAuthenticated, async (req, res) => {
    try {
      const symbol = decodeURIComponent(req.params.symbol);
      const exchange = new ccxt.coinbase({ enableRateLimit: true, timeout: 15000 });
      const ohlcv = await exchange.fetchOHLCV(symbol, '1h', undefined, 50);
      const closes = ohlcv.map(c => c[4] || 0);
      
      const calculateRSI = (prices: number[], period: number = 14) => {
        const rsiValues: { time: string; rsi: number }[] = [];
        for (let i = period; i < prices.length; i++) {
          let gains = 0, losses = 0;
          for (let j = i - period + 1; j <= i; j++) {
            const diff = prices[j] - prices[j - 1];
            if (diff >= 0) gains += diff;
            else losses -= diff;
          }
          const rs = losses === 0 ? 100 : gains / losses;
          const rsi = 100 - (100 / (1 + rs));
          rsiValues.push({
            time: new Date(ohlcv[i][0] || 0).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            rsi
          });
        }
        return rsiValues;
      };
      
      res.json(calculateRSI(closes));
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  app.get("/api/market/:symbol/ticker", isAuthenticated, async (req, res) => {
    try {
      const symbol = decodeURIComponent(req.params.symbol);
      const exchangeParam = req.query.exchange as string | undefined;
      const botId = req.query.botId ? parseInt(req.query.botId as string) : undefined;
      
      let exchangeName = exchangeParam || 'coinbase';
      
      // If botId provided, use the bot's configured exchange
      if (botId) {
        const bot = await storage.getBot(botId);
        if (bot) {
          exchangeName = bot.exchange || 'coinbase';
        }
      }
      
      let exchange: any;
      if (exchangeName === 'kraken') {
        exchange = new ccxt.kraken({ enableRateLimit: true, timeout: 10000 });
      } else {
        exchange = new ccxt.coinbase({ enableRateLimit: true, timeout: 10000 });
      }
      
      const ticker = await exchange.fetchTicker(symbol);
      res.json({
        price: ticker.last,
        change24h: ticker.percentage,
        high24h: ticker.high,
        low24h: ticker.low,
        volume24h: ticker.quoteVolume
      });
    } catch (e: any) {
      // If first exchange fails, try the other as fallback
      try {
        const symbol = decodeURIComponent(req.params.symbol);
        const fallbackExchange = new ccxt.coinbase({ enableRateLimit: true, timeout: 10000 });
        const ticker = await fallbackExchange.fetchTicker(symbol);
        res.json({
          price: ticker.last,
          change24h: ticker.percentage,
          high24h: ticker.high,
          low24h: ticker.low,
          volume24h: ticker.quoteVolume
        });
      } catch (fallbackErr: any) {
        res.status(500).json({ message: e.message });
      }
    }
  });

  app.get("/api/bot/:id/explain-trade/:tradeId", isAuthenticated, async (req: any, res) => {
    try {
      const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
      if (!bot) return;
      const tradeId = parseInt(req.params.tradeId);
      
      const trades = await storage.getTrades(bot.id);
      const trade = trades.find(t => t.id === tradeId);
      if (!trade) {
        return res.status(404).json({ message: "Trade not found" });
      }
      
      const completion = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages: [
          {
            role: "system",
            content: `You are an expert crypto trading analyst. Explain trading decisions in clear, educational terms. Provide detailed technical and sentiment analysis. Return JSON with:
- summary: 1-2 sentence trade summary
- technicalFactors: array of {name, value, impact ('positive'/'negative'/'neutral'), weight (0-1)}
- sentimentFactors: array of {name, value, impact}
- aiReasoning: detailed 2-3 sentence AI reasoning
- confidence: 0-1 confidence level
- riskAssessment: 'Low'/'Medium'/'High'
- expectedOutcome: 'Bullish'/'Bearish'/'Neutral'`
          },
          {
            role: "user",
            content: `Explain this ${trade.side.toUpperCase()} trade:
Symbol: ${trade.symbol}
Price: $${trade.price}
Amount: ${trade.amount}
Entry Reason: ${trade.entryReason || 'N/A'}
Exit Reason: ${trade.exitReason || 'N/A'}
PnL: $${trade.pnl || 0}
Bot Settings: RSI Threshold: ${bot?.rsiThreshold}, EMA Fast: ${bot?.emaFast}, EMA Slow: ${bot?.emaSlow}
Risk Profile: ${bot?.riskProfile}
Sentiment Score: ${bot?.sentimentScore?.toFixed(2) || 'N/A'}`
          }
        ],
        response_format: { type: "json_object" }
      });

      let explanation;
      try {
        explanation = JSON.parse(completion.choices[0].message.content || '{}');
      } catch {
        explanation = {};
      }
      
      res.json({
        tradeId,
        summary: explanation.summary || 'Trade executed based on strategy parameters',
        technicalFactors: explanation.technicalFactors || [],
        sentimentFactors: explanation.sentimentFactors || [],
        aiReasoning: explanation.aiReasoning || 'Analysis not available',
        confidence: explanation.confidence || 0.5,
        riskAssessment: explanation.riskAssessment || 'Medium',
        expectedOutcome: explanation.expectedOutcome || 'Neutral'
      });
    } catch (e: any) {
      res.status(500).json({ 
        message: e.message,
        tradeId: parseInt(req.params.tradeId),
        summary: 'Unable to generate explanation',
        technicalFactors: [],
        sentimentFactors: [],
        aiReasoning: 'Analysis failed',
        confidence: 0,
        riskAssessment: 'Unknown',
        expectedOutcome: 'Unknown'
      });
    }
  });

  // AI-driven Strategy Suggestions
  app.get("/api/bot/:id/strategy-suggestions", isAuthenticated, async (req: any, res) => {
    try {
      const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
      if (!bot) return;
      const trades = await storage.getTrades(bot.id);

      // Calculate performance metrics - only from SELL trades (where P&L is realized)
      const completedTrades = trades.filter(t => t.side === 'sell');
      const winningTrades = completedTrades.filter(t => (t.pnl || 0) > 0);
      const losingTrades = completedTrades.filter(t => (t.pnl || 0) < 0);
      const winRate = completedTrades.length > 0 ? (winningTrades.length / completedTrades.length) * 100 : 0;
      const totalPnL = completedTrades.reduce((acc, t) => acc + (t.pnl || 0), 0);
      const grossProfit = winningTrades.reduce((acc, t) => acc + (t.pnl || 0), 0);
      const grossLoss = Math.abs(losingTrades.reduce((acc, t) => acc + (t.pnl || 0), 0));
      const avgWin = winningTrades.length > 0 ? grossProfit / winningTrades.length : 0;
      const avgLoss = losingTrades.length > 0 ? grossLoss / losingTrades.length : 0;
      const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
      
      // Analyze trade patterns
      const longTrades = trades.filter(t => t.side === 'buy');
      const shortTrades = trades.filter(t => t.side === 'sell');
      const longWinRate = longTrades.length > 0 ? (longTrades.filter(t => (t.pnl || 0) > 0).length / longTrades.length) * 100 : 0;
      const shortWinRate = shortTrades.length > 0 ? (shortTrades.filter(t => (t.pnl || 0) > 0).length / shortTrades.length) * 100 : 0;

      // Get current market data using configured exchange
      let marketConditions = "Unknown";
      let currentPrice = 0;
      let change24h = 0;
      try {
        const exchangeName = bot.exchange || 'coinbase';
        const exchange = exchangeName === 'kraken' 
          ? new ccxt.kraken({ enableRateLimit: true, timeout: 15000 }) 
          : new ccxt.coinbase({ enableRateLimit: true, timeout: 15000 });
        const ticker = await exchange.fetchTicker(bot.symbol);
        currentPrice = ticker.last || 0;
        change24h = ticker.percentage || 0;
        if (change24h > 3) marketConditions = "Bullish";
        else if (change24h < -3) marketConditions = "Bearish";
        else marketConditions = "Ranging";
      } catch (e) {
        // Use neutral if market data unavailable
      }

      const prompt = `You are an expert crypto trading strategist. Analyze the following trading bot performance and provide personalized strategy suggestions.

CURRENT BOT CONFIGURATION:
- Trading Pair: ${bot.symbol}
- Exchange: ${bot.exchange}
- Risk Profile: ${bot.riskProfile || 'moderate'}
- RSI Threshold: ${bot.rsiThreshold || 45}
- EMA Fast: ${bot.emaFast || 9}
- EMA Slow: ${bot.emaSlow || 21}
- Trailing Stop: ${bot.trailingStop ? 'Enabled' : 'Disabled'}
- Current RSI: ${bot.currentRsi?.toFixed(1) || 'N/A'}

HISTORICAL PERFORMANCE:
- Total Trades: ${trades.length}
- Win Rate: ${winRate.toFixed(1)}%
- Total P&L: $${totalPnL.toFixed(2)}
- Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}
- Average Win: $${avgWin.toFixed(2)}
- Average Loss: $${avgLoss.toFixed(2)}
- Long Trade Win Rate: ${longWinRate.toFixed(1)}%
- Short Trade Win Rate: ${shortWinRate.toFixed(1)}%

CURRENT MARKET CONDITIONS:
- Current Price: $${currentPrice.toLocaleString()}
- 24h Change: ${change24h.toFixed(2)}%
- Market Trend: ${marketConditions}

Provide 3-5 specific, actionable strategy suggestions in JSON format:
{
  "marketOutlook": "Brief analysis of current market conditions and outlook",
  "riskAssessment": "Key risk factors and assessment based on performance",
  "suggestions": [
    {
      "title": "Suggestion title",
      "description": "Detailed explanation",
      "impact": "high" | "medium" | "low",
      "category": "timing" | "risk" | "entry" | "exit" | "strategy",
      "action": "Specific parameter change or action to take",
      "confidence": 0.0-1.0 (how confident in this suggestion)
    }
  ],
  "overallScore": 0-100 (current strategy effectiveness score)
}

Base suggestions on actual performance data. If win rate is low, suggest improvements. If long trades outperform shorts, suggest focusing on long positions. Consider current market conditions.`;

      const completion = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages: [{ role: "user", content: prompt }],
        max_completion_tokens: 1500,
        response_format: { type: "json_object" }
      });

      const responseText = completion.choices[0].message.content || "{}";
      let aiResponse;
      try {
        aiResponse = JSON.parse(responseText);
      } catch (parseError) {
        aiResponse = {
          marketOutlook: "Unable to parse AI response",
          riskAssessment: "Analysis unavailable",
          suggestions: [],
          overallScore: 50
        };
      }
      
      // Calculate streak data
      let currentStreak = 0;
      let maxWinStreak = 0;
      let maxLossStreak = 0;
      let tempWinStreak = 0;
      let tempLossStreak = 0;
      
      const sortedTrades = [...trades].sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      
      for (const trade of sortedTrades) {
        if ((trade.pnl || 0) > 0) {
          tempWinStreak++;
          tempLossStreak = 0;
          currentStreak = tempWinStreak;
          maxWinStreak = Math.max(maxWinStreak, tempWinStreak);
        } else if ((trade.pnl || 0) < 0) {
          tempLossStreak++;
          tempWinStreak = 0;
          currentStreak = -tempLossStreak;
          maxLossStreak = Math.max(maxLossStreak, tempLossStreak);
        }
      }
      
      // Analyze best/worst trading hours
      const hourPnL: Record<number, { wins: number; losses: number; pnl: number }> = {};
      for (const trade of trades) {
        const hour = new Date(trade.timestamp).getHours();
        if (!hourPnL[hour]) hourPnL[hour] = { wins: 0, losses: 0, pnl: 0 };
        hourPnL[hour].pnl += trade.pnl || 0;
        if ((trade.pnl || 0) > 0) hourPnL[hour].wins++;
        else if ((trade.pnl || 0) < 0) hourPnL[hour].losses++;
      }
      
      const hourEntries = Object.entries(hourPnL).map(([h, d]) => ({ 
        hour: parseInt(h), 
        ...d, 
        winRate: d.wins + d.losses > 0 ? d.wins / (d.wins + d.losses) : 0 
      }));
      const bestHours = hourEntries.filter(h => h.winRate >= 0.6 && h.wins + h.losses >= 2).map(h => h.hour).slice(0, 3);
      const worstHours = hourEntries.filter(h => h.winRate <= 0.4 && h.wins + h.losses >= 2).map(h => h.hour).slice(0, 3);
      
      // Ensure suggestions have confidence values
      const suggestions = (aiResponse.suggestions || []).map((s: any) => ({
        ...s,
        confidence: s.confidence || 0.7
      }));
      
      res.json({
        suggestions,
        marketOutlook: aiResponse.marketOutlook || "Market conditions unavailable",
        riskAssessment: aiResponse.riskAssessment || "Risk assessment unavailable",
        metrics: {
          totalTrades: trades.length,
          winRate,
          totalPnL,
          avgWin,
          avgLoss,
          profitFactor: profitFactor === Infinity ? 999 : profitFactor,
          currentRsi: bot.currentRsi || 50,
          maxWinStreak,
          maxLossStreak,
          currentStreak,
          bestHours,
          worstHours
        },
        modeInfo: {
          isLiveMode: bot.isLiveMode || false,
          isLiveAnalysis: bot.isLiveMode || false,
          liveTradeCount: bot.isLiveMode ? trades.length : 0,
          liveWinRate: bot.isLiveMode ? winRate : 0,
          livePnL: bot.isLiveMode ? totalPnL : 0,
          paperTradeCount: !bot.isLiveMode ? trades.length : 0,
          paperWinRate: !bot.isLiveMode ? winRate : 0,
          paperPnL: !bot.isLiveMode ? totalPnL : 0
        },
        generatedAt: new Date().toISOString()
      });
    } catch (e: any) {
      console.error("Strategy suggestions error:", e);
      res.status(500).json({ 
        message: e.message,
        suggestions: [],
        marketOutlook: "Unable to analyze market conditions",
        riskAssessment: "Unable to generate risk assessment",
        metrics: {
          totalTrades: 0,
          winRate: 0,
          totalPnL: 0,
          avgWin: 0,
          avgLoss: 0,
          profitFactor: 0,
          currentRsi: 50,
          maxWinStreak: 0,
          maxLossStreak: 0,
          currentStreak: 0,
          bestHours: [],
          worstHours: []
        },
        generatedAt: new Date().toISOString()
      });
    }
  });

  // ==================== COMMUNITY FORUM API ====================
  
  // Import forum schemas for validation
  const { insertForumCategorySchema, insertForumTopicSchema, insertForumPostSchema } = await import("@shared/schema");

  // Get all forum categories
  app.get("/api/forum/categories", async (req, res) => {
    try {
      const categories = await storage.getForumCategories();
      if (categories.length === 0) {
        const defaultCategories = [
          { name: "Trading Strategies", description: "Discuss and share trading strategies", icon: "TrendingUp", sortOrder: 1 },
          { name: "Market Analysis", description: "Share market insights and analysis", icon: "BarChart3", sortOrder: 2 },
          { name: "Bot Configuration", description: "Get help configuring your trading bot", icon: "Settings", sortOrder: 3 },
          { name: "General Discussion", description: "Off-topic conversations and community chat", icon: "MessageCircle", sortOrder: 4 },
          { name: "Beginner Questions", description: "New to crypto trading? Ask here!", icon: "HelpCircle", sortOrder: 5 },
        ];
        for (const cat of defaultCategories) {
          await storage.createForumCategory(cat);
        }
        const seeded = await storage.getForumCategories();
        return res.json(seeded);
      }
      res.json(categories);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  // Create new forum category (admin only in production)
  app.post("/api/forum/categories", async (req, res) => {
    try {
      const validated = insertForumCategorySchema.parse(req.body);
      const category = await storage.createForumCategory(validated);
      res.json(category);
    } catch (e: any) {
      if (e.name === 'ZodError') {
        return res.status(400).json({ message: "Validation error", errors: e.errors });
      }
      res.status(500).json({ message: e.message });
    }
  });

  // Get all topics (optionally filtered by category)
  app.get("/api/forum/topics", async (req, res) => {
    try {
      const categoryId = req.query.categoryId ? parseInt(req.query.categoryId as string) : undefined;
      const topics = await storage.getForumTopics(categoryId);
      res.json(topics);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  // Get single topic by ID
  app.get("/api/forum/topics/:id", async (req, res) => {
    try {
      const topic = await storage.getForumTopic(parseInt(req.params.id));
      if (!topic) return res.status(404).json({ message: "Topic not found" });
      res.json(topic);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  // Create new topic
  app.post("/api/forum/topics", isAuthenticated, async (req: any, res) => {
    try {
      const user = req.user?.claims;
      const authorName = user?.first_name 
        ? `${user.first_name}${user.last_name ? ' ' + user.last_name : ''}`
        : user?.email?.split('@')[0] || "Anonymous";
      
      const validated = insertForumTopicSchema.parse({
        categoryId: req.body.categoryId,
        title: req.body.title,
        content: req.body.content,
        authorId: user?.sub,
        authorName: authorName,
        authorAvatar: user?.profile_image_url || req.body.authorAvatar,
        isPinned: req.body.isPinned || false,
        isLocked: req.body.isLocked || false,
        tags: req.body.tags || [],
      });
      const topic = await storage.createForumTopic(validated);
      res.json(topic);
    } catch (e: any) {
      if (e.name === 'ZodError') {
        return res.status(400).json({ message: "Validation error", errors: e.errors });
      }
      res.status(500).json({ message: e.message });
    }
  });

  // Upvote a topic
  app.post("/api/forum/topics/:id/upvote", async (req, res) => {
    try {
      const topic = await storage.upvoteForumTopic(parseInt(req.params.id));
      if (!topic) return res.status(404).json({ message: "Topic not found" });
      res.json(topic);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  // Get posts for a topic
  app.get("/api/forum/topics/:id/posts", async (req, res) => {
    try {
      const posts = await storage.getForumPosts(parseInt(req.params.id));
      res.json(posts);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  // Create new post (reply)
  app.post("/api/forum/topics/:id/posts", isAuthenticated, async (req: any, res) => {
    try {
      const user = req.user?.claims;
      const authorName = user?.first_name 
        ? `${user.first_name}${user.last_name ? ' ' + user.last_name : ''}`
        : user?.email?.split('@')[0] || "Anonymous";
      
      const validated = insertForumPostSchema.parse({
        topicId: parseInt(req.params.id),
        content: req.body.content,
        authorId: user?.sub,
        authorName: authorName,
        authorAvatar: user?.profile_image_url || req.body.authorAvatar,
        isAnswer: req.body.isAnswer || false,
      });
      const post = await storage.createForumPost(validated);
      res.json(post);
    } catch (e: any) {
      if (e.name === 'ZodError') {
        return res.status(400).json({ message: "Validation error", errors: e.errors });
      }
      res.status(500).json({ message: e.message });
    }
  });

  // Upvote a post
  app.post("/api/forum/posts/:id/upvote", async (req, res) => {
    try {
      const post = await storage.upvoteForumPost(parseInt(req.params.id));
      if (!post) return res.status(404).json({ message: "Post not found" });
      res.json(post);
    } catch (e: any) {
      res.status(500).json({ message: e.message });
    }
  });

  // Crypto News endpoint - AI-generated current market news
  let newsCache: { data: any[]; timestamp: Date } = { data: [], timestamp: new Date(0) };
  const NEWS_CACHE_TTL_MS = 300000; // 5 minutes cache

  app.get("/api/news", async (req, res) => {
    try {
      const now = new Date();
      if (newsCache.data.length > 0 && now.getTime() - newsCache.timestamp.getTime() < NEWS_CACHE_TTL_MS) {
        return res.json(newsCache.data);
      }

      const bot = await storage.getBot(1);
      const currentSymbol = bot?.symbol || 'BTC/USDT';
      const baseAsset = currentSymbol.split('/')[0];

      const prompt = `Generate 10 realistic cryptocurrency news headlines and summaries for today's market. Include a mix of:
- Bitcoin and Ethereum news
- ${baseAsset} specific news if not BTC/ETH
- DeFi and altcoin developments
- Regulatory news
- Market adoption stories
- Technology updates

For each article, provide JSON with these fields:
- id: unique string
- title: headline (50-80 chars)
- summary: 2-3 sentence summary
- sentiment: "bullish", "bearish", or "neutral"
- category: "market", "regulation", "technology", "defi", or "adoption"
- assets: array of relevant crypto symbols (e.g., ["BTC", "ETH"])
- source: realistic news source name
- importance: "high", "medium", or "low"

Return ONLY valid JSON array, no markdown.`;

      const response = await openai.chat.completions.create({
        model: "gpt-5.2",
        messages: [{ role: "user", content: prompt }],
        max_completion_tokens: 2000
      });

      const content = response.choices[0]?.message?.content || '[]';
      let articles: any[] = [];
      
      try {
        const cleaned = content.replace(/```json\n?|\n?```/g, '').trim();
        articles = JSON.parse(cleaned);
      } catch {
        articles = [];
      }

      const now2 = new Date();
      const newsWithTimestamps = articles.map((article: any, index: number) => ({
        ...article,
        id: article.id || `news-${Date.now()}-${index}`,
        timestamp: new Date(now2.getTime() - index * 15 * 60000).toISOString() // Stagger by 15 min
      }));

      newsCache = { data: newsWithTimestamps, timestamp: now };
      res.json(newsWithTimestamps);
    } catch (e: any) {
      console.error("News generation error:", e);
      // Return fallback news on error
      res.json([
        {
          id: "fallback-1",
          title: "Cryptocurrency Markets Show Mixed Signals Amid Global Uncertainty",
          summary: "Major cryptocurrencies are trading sideways as investors await key economic data. Bitcoin holds above key support while altcoins show varied performance.",
          sentiment: "neutral",
          category: "market",
          assets: ["BTC", "ETH"],
          source: "CryptoNews",
          importance: "medium",
          timestamp: new Date().toISOString()
        },
        {
          id: "fallback-2",
          title: "DeFi Protocols See Increased Activity as New Features Launch",
          summary: "Decentralized finance platforms report growing user engagement following protocol upgrades. Total value locked continues to climb across major chains.",
          sentiment: "bullish",
          category: "defi",
          assets: ["ETH", "AAVE", "UNI"],
          source: "DeFi Daily",
          importance: "medium",
          timestamp: new Date(Date.now() - 30 * 60000).toISOString()
        }
      ]);
    }
  });

  // ============================================
  // ADVANCED ORDERS API - OCO, Grid, Pending Orders
  // ============================================

  // Get all pending orders for a bot
  app.get("/api/bot/:id/orders/pending", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const orders = await storage.getPendingOrders(bot.id);
      res.json(orders);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Create a new pending order (limit, stop, OCO, trailing stop)
  app.post("/api/bot/:id/orders/pending", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const { orderType, symbol, side, amount, triggerPrice, limitPrice, stopPrice, takeProfitPrice, trailingPercent, expiresAt, notes } = req.body;
      
      // Validate required fields
      if (!orderType || !symbol || !side || !amount) {
        return res.status(400).json({ message: "Missing required fields: orderType, symbol, side, amount" });
      }
      
      // For OCO orders, create two linked orders
      if (orderType === 'oco') {
        if (!takeProfitPrice || !stopPrice) {
          return res.status(400).json({ message: "OCO orders require both takeProfitPrice and stopPrice" });
        }
        
        // Create take profit order
        const tpOrder = await storage.createPendingOrder({
          botId: bot.id,
          userId: req.user.id,
          symbol,
          orderType: 'oco',
          side: 'sell',
          amount,
          triggerPrice: takeProfitPrice,
          takeProfitPrice,
          stopPrice,
          notes: notes || 'OCO Take Profit'
        });
        
        // Create stop loss order, linked to TP
        const slOrder = await storage.createPendingOrder({
          botId: bot.id,
          userId: req.user.id,
          symbol,
          orderType: 'oco',
          side: 'sell',
          amount,
          triggerPrice: stopPrice,
          stopPrice,
          takeProfitPrice,
          linkedOrderId: tpOrder.id,
          notes: notes || 'OCO Stop Loss'
        });
        
        // Update TP order with linked order ID
        await storage.updatePendingOrder(tpOrder.id, { linkedOrderId: slOrder.id });
        
        await storage.createLog({
          botId: bot.id,
          level: 'info',
          message: `[OCO] Created: TP at $${takeProfitPrice}, SL at $${stopPrice} for ${amount} ${symbol}`
        });
        
        return res.json({ takeProfit: tpOrder, stopLoss: slOrder });
      }
      
      // Regular pending order
      const order = await storage.createPendingOrder({
        botId: bot.id,
        userId: req.user.id,
        symbol,
        orderType,
        side,
        amount,
        triggerPrice,
        limitPrice,
        stopPrice,
        takeProfitPrice,
        trailingPercent,
        expiresAt: expiresAt ? new Date(expiresAt) : undefined,
        notes
      });
      
      await storage.createLog({
        botId: bot.id,
        level: 'info',
        message: `[ORDER] Created ${orderType} ${side} order: ${amount} ${symbol} @ $${triggerPrice || limitPrice || 'market'}`
      });
      
      res.json(order);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Cancel a pending order
  app.delete("/api/bot/:id/orders/pending/:orderId", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const orderId = parseInt(req.params.orderId);
      const orders = await storage.getPendingOrders(bot.id);
      const order = orders.find(o => o.id === orderId);
      
      if (!order) {
        return res.status(404).json({ message: "Order not found" });
      }
      
      // For OCO orders, also cancel the linked order
      if (order.linkedOrderId) {
        await storage.updatePendingOrder(order.linkedOrderId, { status: 'cancelled' });
      }
      
      await storage.updatePendingOrder(orderId, { status: 'cancelled' });
      
      await storage.createLog({
        botId: bot.id,
        level: 'info',
        message: `[ORDER] Cancelled ${order.orderType} order #${orderId}`
      });
      
      res.json({ success: true });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // ============================================
  // GRID TRADING API
  // ============================================

  // Get all grid configs for a bot
  app.get("/api/bot/:id/grids", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const grids = await storage.getGridConfigs(bot.id);
      res.json(grids);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Create a new grid trading configuration
  app.post("/api/bot/:id/grids", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const { symbol, gridType, upperPrice, lowerPrice, gridLevels, amountPerGrid, totalInvestment } = req.body;
      
      if (!symbol || !upperPrice || !lowerPrice || !gridLevels || !totalInvestment) {
        return res.status(400).json({ message: "Missing required fields" });
      }
      
      if (upperPrice <= lowerPrice) {
        return res.status(400).json({ message: "Upper price must be greater than lower price" });
      }
      
      const grid = await storage.createGridConfig({
        botId: bot.id,
        userId: req.user.id,
        symbol,
        gridType: gridType || 'arithmetic',
        upperPrice,
        lowerPrice,
        gridLevels,
        amountPerGrid: amountPerGrid || (totalInvestment / gridLevels),
        totalInvestment
      });
      
      // Create grid orders
      const priceStep = gridType === 'geometric' 
        ? Math.pow(upperPrice / lowerPrice, 1 / (gridLevels - 1))
        : (upperPrice - lowerPrice) / (gridLevels - 1);
      
      for (let i = 0; i < gridLevels; i++) {
        const price = gridType === 'geometric'
          ? lowerPrice * Math.pow(priceStep, i)
          : lowerPrice + (priceStep * i);
        
        // Create buy order at each grid level
        await storage.createPendingOrder({
          botId: bot.id,
          userId: req.user.id,
          symbol,
          orderType: 'grid',
          side: 'buy',
          amount: grid.amountPerGrid,
          triggerPrice: price,
          gridId: grid.id,
          notes: `Grid level ${i + 1}/${gridLevels}`
        });
      }
      
      await storage.createLog({
        botId: bot.id,
        level: 'success',
        message: `[GRID] Created ${gridLevels}-level grid for ${symbol}: $${lowerPrice.toFixed(2)} - $${upperPrice.toFixed(2)}`
      });
      
      res.json(grid);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Start/Stop a grid
  app.patch("/api/bot/:id/grids/:gridId", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const gridId = parseInt(req.params.gridId);
      const { status } = req.body;
      
      const grid = await storage.getGridConfig(gridId);
      if (!grid || grid.botId !== bot.id) {
        return res.status(404).json({ message: "Grid not found" });
      }
      
      const update: any = { status };
      if (status === 'running' && !grid.startedAt) {
        update.startedAt = new Date();
      } else if (status === 'stopped') {
        update.stoppedAt = new Date();
      }
      
      const updated = await storage.updateGridConfig(gridId, update);
      
      await storage.createLog({
        botId: bot.id,
        level: 'info',
        message: `[GRID] ${grid.symbol} grid ${status === 'running' ? 'started' : 'stopped'}`
      });
      
      res.json(updated);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Delete a grid and its orders
  app.delete("/api/bot/:id/grids/:gridId", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const gridId = parseInt(req.params.gridId);
      
      // Cancel all grid orders
      const orders = await storage.getPendingOrders(bot.id);
      for (const order of orders.filter(o => o.gridId === gridId && o.status === 'pending')) {
        await storage.updatePendingOrder(order.id, { status: 'cancelled' });
      }
      
      await storage.deleteGridConfig(gridId);
      
      await storage.createLog({
        botId: bot.id,
        level: 'info',
        message: `[GRID] Deleted grid #${gridId} and cancelled associated orders`
      });
      
      res.json({ success: true });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // ============================================
  // PORTFOLIO REBALANCING API
  // ============================================

  // Get portfolio allocations
  app.get("/api/bot/:id/portfolio/allocations", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const allocations = await storage.getPortfolioAllocations(bot.id);
      res.json(allocations);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Set target allocation for a symbol
  app.post("/api/bot/:id/portfolio/allocations", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const { symbol, targetPercent, rebalanceThreshold } = req.body;
      
      if (!symbol || targetPercent === undefined) {
        return res.status(400).json({ message: "Missing symbol or targetPercent" });
      }
      
      // Check if allocation already exists
      const existing = await storage.getPortfolioAllocations(bot.id);
      const found = existing.find(a => a.symbol === symbol);
      
      if (found) {
        const updated = await storage.updatePortfolioAllocation(found.id, { 
          targetPercent, 
          rebalanceThreshold: rebalanceThreshold || 5 
        });
        return res.json(updated);
      }
      
      const allocation = await storage.createPortfolioAllocation({
        botId: bot.id,
        userId: req.user.id,
        symbol,
        targetPercent,
        rebalanceThreshold: rebalanceThreshold || 5
      });
      
      res.json(allocation);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Delete a portfolio allocation
  app.delete("/api/bot/:id/portfolio/allocations/:allocId", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      await storage.deletePortfolioAllocation(parseInt(req.params.allocId));
      res.json({ success: true });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Get rebalancing schedule
  app.get("/api/bot/:id/portfolio/schedule", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const schedule = await storage.getRebalanceSchedule(bot.id);
      res.json(schedule || { scheduleType: 'manual', isEnabled: false });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Update rebalancing schedule
  app.post("/api/bot/:id/portfolio/schedule", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      const { scheduleType, isEnabled, thresholdPercent } = req.body;
      
      let schedule = await storage.getRebalanceSchedule(bot.id);
      
      if (schedule) {
        schedule = await storage.updateRebalanceSchedule(schedule.id, {
          scheduleType,
          isEnabled,
          thresholdPercent
        });
      } else {
        schedule = await storage.createRebalanceSchedule({
          botId: bot.id,
          userId: req.user.id,
          scheduleType: scheduleType || 'manual',
          isEnabled: isEnabled || false,
          thresholdPercent: thresholdPercent || 5
        });
      }
      
      res.json(schedule);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Execute manual rebalance
  app.post("/api/bot/:id/portfolio/rebalance", isAuthenticated, async (req: any, res) => {
    const bot = await verifyBotOwnership(req, res, parseInt(req.params.id));
    if (!bot) return;
    
    try {
      if (!bot.isLiveMode) {
        return res.status(400).json({ message: "Rebalancing only available in live mode" });
      }
      
      const allocations = await storage.getPortfolioAllocations(bot.id);
      if (allocations.length === 0) {
        return res.status(400).json({ message: "No target allocations configured" });
      }
      
      // This would trigger the actual rebalancing logic in the engine
      await storage.createLog({
        botId: bot.id,
        level: 'info',
        message: `[REBALANCE] Manual rebalance triggered for ${allocations.length} assets`
      });
      
      // Return current vs target allocations for preview
      res.json({
        message: "Rebalance analysis initiated",
        allocations: allocations.map(a => ({
          symbol: a.symbol,
          targetPercent: a.targetPercent,
          currentPercent: a.currentPercent || 0,
          deviation: Math.abs((a.currentPercent || 0) - a.targetPercent)
        }))
      });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // GitHub Integration - Push code to GitHub
  app.get("/api/github/user", isAuthenticated, async (req: any, res) => {
    try {
      const user = await getGitHubUser();
      res.json(user);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  app.post("/api/github/push", isAuthenticated, async (req: any, res) => {
    try {
      const { repoName, isPrivate = true } = req.body;
      const user = await getGitHubUser();
      const owner = user.login;
      const finalRepoName = repoName || "astraeus-ai-trading-bot";
      
      // Check if repo exists, create if not
      let repo = await getRepo(owner, finalRepoName);
      if (!repo) {
        repo = await createRepo(finalRepoName, "AI-powered crypto trading bot built with Astraeus AI", isPrivate);
      }

      // Files to push (key project files)
      const filesToPush = [
        "package.json",
        "tsconfig.json",
        "vite.config.ts",
        "drizzle.config.ts",
        "tailwind.config.ts",
        "README.md",
        "client/index.html",
        "client/src/App.tsx",
        "client/src/main.tsx",
        "shared/schema.ts",
        "server/index.ts",
        "server/routes.ts",
        "server/storage.ts",
        "server/engine.ts",
      ];

      const pushedFiles: string[] = [];
      const errors: string[] = [];

      for (const filePath of filesToPush) {
        try {
          const fullPath = path.join(process.cwd(), filePath);
          if (fs.existsSync(fullPath)) {
            const content = fs.readFileSync(fullPath, "utf-8");
            
            // Check if file exists on GitHub to get SHA
            const existing = await getFileContent(owner, finalRepoName, filePath);
            const sha = existing && "sha" in existing ? existing.sha : undefined;
            
            await pushFile(owner, finalRepoName, filePath, content, `Update ${filePath}`, sha);
            pushedFiles.push(filePath);
          }
        } catch (e: any) {
          errors.push(`${filePath}: ${e.message}`);
        }
      }

      res.json({
        success: true,
        repoUrl: `https://github.com/${owner}/${finalRepoName}`,
        pushedFiles,
        errors: errors.length > 0 ? errors : undefined,
      });
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  return httpServer;
}
