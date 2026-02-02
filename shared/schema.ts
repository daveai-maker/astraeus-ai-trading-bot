import { pgTable, text, serial, integer, boolean, doublePrecision, timestamp, jsonb, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { sql } from "drizzle-orm";
import { users } from "./models/auth";

export * from "./models/auth";

export const bots = pgTable("bots", {
  id: serial("id").primaryKey(),
  userId: text("user_id").references(() => users.id),
  name: text("name").notNull().default("Astraeus AI"),
  isRunning: boolean("is_running").notNull().default(false),
  isLiveMode: boolean("is_live_mode").notNull().default(false),
  exchange: text("exchange").notNull().default("coinbase"),
  symbol: text("symbol").notNull().default("BTC/USDT"),
  intervalSeconds: integer("interval_seconds").notNull().default(60),
  lastError: text("last_error"),
  lastRun: timestamp("last_run"),
  coinbaseApiKey: text("coinbase_api_key"),
  coinbaseApiSecret: text("coinbase_api_secret"),
  krakenApiKey: text("kraken_api_key"),
  krakenApiSecret: text("kraken_api_secret"),
  riskProfile: text("risk_profile").notNull().default("safe"), // 'safe', 'balanced', 'aggressive'
  drawdown: doublePrecision("drawdown").default(0),
  profitFactor: doublePrecision("profit_factor").default(0),
  slippage: doublePrecision("slippage").default(0),
  fillRate: doublePrecision("fill_rate").default(0),
  latency: integer("latency").default(0),
  errorRate: doublePrecision("error_rate").default(0),
  sentimentScore: doublePrecision("sentiment_score"),
  isPaused: boolean("is_paused").default(false),
  equityHistory: jsonb("equity_history").$type<{timestamp: string, balance: number}[]>().default([]),
  telegramWebhook: text("telegram_webhook"),
  discordWebhook: text("discord_webhook"),
  rsiThreshold: integer("rsi_threshold").default(45),
  emaFast: integer("ema_fast").default(9),
  emaSlow: integer("ema_slow").default(21),
  trailingStop: boolean("trailing_stop").default(false),
  highestPrice: doublePrecision("highest_price"),
  currentRsi: doublePrecision("current_rsi"),
  currentEmaFast: doublePrecision("current_ema_fast"),
  currentEmaSlow: doublePrecision("current_ema_slow"),
  sharpeRatio: doublePrecision("sharpe_ratio").default(0),
  sortinoRatio: doublePrecision("sortino_ratio").default(0),
  totalReturn: doublePrecision("total_return").default(0),
  avgTradeSize: doublePrecision("avg_trade_size").default(0),
  aiConfidence: doublePrecision("ai_confidence").default(0),
  aiReasoning: text("ai_reasoning"),
  lastSignal: text("last_signal"),
  watchlist: jsonb("watchlist").$type<string[]>().default(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
  // Paper trading enhancements
  paperStartingCapital: doublePrecision("paper_starting_capital").default(10000),
  paperBalance: doublePrecision("paper_balance").default(10000),
  paperTotalFees: doublePrecision("paper_total_fees").default(0),
  paperTotalSlippage: doublePrecision("paper_total_slippage").default(0),
  simulatedFeeRate: doublePrecision("simulated_fee_rate").default(0.001), // 0.1% default fee
  simulatedSlippageRate: doublePrecision("simulated_slippage_rate").default(0.0005), // 0.05% default slippage
  paperWinCount: integer("paper_win_count").default(0),
  paperLossCount: integer("paper_loss_count").default(0),
  paperBestTrade: doublePrecision("paper_best_trade").default(0),
  paperWorstTrade: doublePrecision("paper_worst_trade").default(0),
  paperStartedAt: timestamp("paper_started_at"),
  // Live trading safety limits
  maxOrderSize: doublePrecision("max_order_size").default(500),
  dailyLossLimit: doublePrecision("daily_loss_limit").default(100),
  maxDailyTrades: integer("max_daily_trades").default(20),
  // Persisted daily trading stats (reset at midnight)
  dailyStatsDate: text("daily_stats_date"),
  dailyTotalPnL: doublePrecision("daily_total_pnl").default(0),
  dailyTradeCount: integer("daily_trade_count").default(0),
  // Live equity history (separate from paper)
  liveEquityHistory: jsonb("live_equity_history").$type<{timestamp: string, balance: number}[]>().default([]),
  // Enhanced Live Trading Features
  useSmartOrders: boolean("use_smart_orders").default(false), // Use limit orders with smart placement
  smartOrderSpread: doublePrecision("smart_order_spread").default(0.001), // 0.1% spread for limit orders
  dcaEnabled: boolean("dca_enabled").default(false), // Dollar-cost averaging mode
  dcaInterval: integer("dca_interval").default(3600), // DCA interval in seconds (1 hour default)
  dcaAmount: doublePrecision("dca_amount").default(10), // DCA amount per interval
  dcaLastBuy: timestamp("dca_last_buy"), // Track last DCA purchase
  multiAssetEnabled: boolean("multi_asset_enabled").default(false), // Trade strongest from watchlist
  volatilityScaling: boolean("volatility_scaling").default(true), // ATR-based position sizing
  trailingStopPercent: doublePrecision("trailing_stop_percent").default(2), // Trailing stop % from high
  trailingStopActive: boolean("trailing_stop_active").default(false), // Is trailing stop armed
  trailingHighWaterMark: doublePrecision("trailing_high_water_mark"), // Highest price since entry
  killSwitchEnabled: boolean("kill_switch_enabled").default(false), // Emergency halt trading
  maxDrawdownPercent: doublePrecision("max_drawdown_percent").default(10), // Max drawdown before auto-halt
  peakEquity: doublePrecision("peak_equity"), // Peak equity for drawdown calculation
  orderRetryAttempts: integer("order_retry_attempts").default(3), // Retry failed orders
  lastOrderError: text("last_order_error"), // Track order failures
  // Strategy Configuration
  strategyType: text("strategy_type").default("adaptive"), // adaptive, scalping, swing, mean_reversion, trend_following
  // Technical Indicator Settings
  rsiOversold: integer("rsi_oversold").default(30), // RSI oversold threshold
  rsiOverbought: integer("rsi_overbought").default(70), // RSI overbought threshold
  macdFast: integer("macd_fast").default(12), // MACD fast period
  macdSlow: integer("macd_slow").default(26), // MACD slow period
  macdSignal: integer("macd_signal").default(9), // MACD signal period
  bollingerPeriod: integer("bollinger_period").default(20), // Bollinger Bands period
  bollingerStdDev: doublePrecision("bollinger_std_dev").default(2), // Bollinger Bands std dev
  atrPeriod: integer("atr_period").default(14), // ATR period for volatility
  atrMultiplier: doublePrecision("atr_multiplier").default(1.5), // ATR multiplier for stops
  // Entry Rules
  minEntryScore: integer("min_entry_score").default(4), // Minimum score to enter trade
  minAiConfidence: doublePrecision("min_ai_confidence").default(0.6), // Minimum AI confidence (0-1)
  requireVolumeConfirm: boolean("require_volume_confirm").default(true), // Require volume surge for entry
  requireTrendAlign: boolean("require_trend_align").default(true), // Require trend alignment
  // Exit Rules
  profitTargetPercent: doublePrecision("profit_target_percent").default(3), // Take profit target %
  stopLossPercent: doublePrecision("stop_loss_percent").default(2), // Stop loss %
  useAiExitSignals: boolean("use_ai_exit_signals").default(true), // Use AI for exit decisions
  // Market Filters
  minVolatility: doublePrecision("min_volatility").default(0.5), // Min volatility % to trade
  maxVolatility: doublePrecision("max_volatility").default(10), // Max volatility % to trade
  tradingHoursOnly: boolean("trading_hours_only").default(false), // Only trade during peak hours
  avoidWeekends: boolean("avoid_weekends").default(false), // Skip weekend trading
  // AI Settings
  aiAggressiveness: text("ai_aggressiveness").default("balanced"), // conservative, balanced, aggressive
  useInstitutionalSignals: boolean("use_institutional_signals").default(true), // VPIN, order flow, etc
  useSentimentAnalysis: boolean("use_sentiment_analysis").default(true), // News/social sentiment
});

export const backtests = pgTable("backtests", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").notNull(),
  symbol: text("symbol").notNull(),
  days: integer("days").notNull(),
  totalTrades: integer("total_trades").notNull(),
  winRate: doublePrecision("win_rate").notNull(),
  netProfit: doublePrecision("net_profit").notNull(),
  maxDrawdown: doublePrecision("max_drawdown").notNull(),
  timestamp: timestamp("timestamp").defaultNow(),
  equityCurve: jsonb("equity_curve").$type<{time: number, value: number}[]>().default([]),
  tradeLog: jsonb("trade_log").$type<{time: number, side: string, price: number, pnl: number}[]>().default([]),
  rsiThreshold: integer("rsi_threshold"),
  emaFast: integer("ema_fast"),
  emaSlow: integer("ema_slow"),
  riskProfile: text("risk_profile"),
  sharpeRatio: doublePrecision("sharpe_ratio"),
  profitFactor: doublePrecision("profit_factor"),
});

export const insertBacktestSchema = createInsertSchema(backtests).omit({ id: true, timestamp: true });
export type InsertBacktest = z.infer<typeof insertBacktestSchema>;
export type Backtest = typeof backtests.$inferSelect;

export const conversations = pgTable("conversations", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
});

export const messages = pgTable("messages", {
  id: serial("id").primaryKey(),
  conversationId: integer("conversation_id").notNull().references(() => conversations.id, { onDelete: "cascade" }),
  role: text("role").notNull(),
  content: text("content").notNull(),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
});

export const trades = pgTable("trades", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").references(() => bots.id).notNull(),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(), 
  price: doublePrecision("price").notNull(),
  amount: doublePrecision("amount").notNull(),
  pnl: doublePrecision("pnl"),
  status: text("status").notNull(), 
  timestamp: timestamp("timestamp").defaultNow().notNull(),
  entryReason: text("entry_reason"),
  exitReason: text("exit_reason"),
  fees: doublePrecision("fees").default(0),
  slippage: doublePrecision("slippage").default(0),
  executedPrice: doublePrecision("executed_price"),
  isPaperTrade: boolean("is_paper_trade").default(true),
  orderId: text("order_id"),
});

export const logs = pgTable("logs", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").references(() => bots.id).notNull(),
  level: text("level").notNull(), 
  message: text("message").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

export const insertBotSchema = createInsertSchema(bots).omit({ id: true, lastRun: true });
export const insertTradeSchema = createInsertSchema(trades).omit({ id: true, timestamp: true });
export const insertLogSchema = createInsertSchema(logs).omit({ id: true, timestamp: true });

export type Bot = typeof bots.$inferSelect;
export type InsertBot = z.infer<typeof insertBotSchema>;
export type Trade = typeof trades.$inferSelect;
export type InsertTrade = z.infer<typeof insertTradeSchema>;
export type Log = typeof logs.$inferSelect;
export type InsertLog = z.infer<typeof insertLogSchema>;

// Community Forum Schema
export const forumCategories = pgTable("forum_categories", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  icon: text("icon").default("MessageSquare"),
  sortOrder: integer("sort_order").default(0),
  topicCount: integer("topic_count").default(0),
});

export const forumTopics = pgTable("forum_topics", {
  id: serial("id").primaryKey(),
  categoryId: integer("category_id").references(() => forumCategories.id).notNull(),
  title: text("title").notNull(),
  content: text("content").notNull(),
  authorId: text("author_id").references(() => users.id),
  authorName: text("author_name").notNull().default("Anonymous"),
  authorAvatar: text("author_avatar"),
  isPinned: boolean("is_pinned").default(false),
  isLocked: boolean("is_locked").default(false),
  viewCount: integer("view_count").default(0),
  replyCount: integer("reply_count").default(0),
  upvotes: integer("upvotes").default(0),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
  updatedAt: timestamp("updated_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
  lastReplyAt: timestamp("last_reply_at"),
  tags: jsonb("tags").$type<string[]>().default([]),
});

export const forumPosts = pgTable("forum_posts", {
  id: serial("id").primaryKey(),
  topicId: integer("topic_id").references(() => forumTopics.id, { onDelete: "cascade" }).notNull(),
  content: text("content").notNull(),
  authorId: text("author_id").references(() => users.id),
  authorName: text("author_name").notNull().default("Anonymous"),
  authorAvatar: text("author_avatar"),
  upvotes: integer("upvotes").default(0),
  isAnswer: boolean("is_answer").default(false),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
  updatedAt: timestamp("updated_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
});

export const insertForumCategorySchema = createInsertSchema(forumCategories).omit({ id: true });
export const insertForumTopicSchema = createInsertSchema(forumTopics).omit({ id: true, createdAt: true, updatedAt: true, lastReplyAt: true, viewCount: true, replyCount: true, upvotes: true });
export const insertForumPostSchema = createInsertSchema(forumPosts).omit({ id: true, createdAt: true, updatedAt: true, upvotes: true });

export type ForumCategory = typeof forumCategories.$inferSelect;
export type InsertForumCategory = z.infer<typeof insertForumCategorySchema>;
export type ForumTopic = typeof forumTopics.$inferSelect;
export type InsertForumTopic = z.infer<typeof insertForumTopicSchema>;
export type ForumPost = typeof forumPosts.$inferSelect;
export type InsertForumPost = z.infer<typeof insertForumPostSchema>;

// Advanced Orders Schema - Pending orders, OCO, Grid Trading
export const pendingOrders = pgTable("pending_orders", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").references(() => bots.id).notNull(),
  userId: text("user_id").references(() => users.id),
  symbol: text("symbol").notNull(),
  orderType: text("order_type").notNull(), // 'limit', 'stop', 'oco', 'trailing_stop', 'grid'
  side: text("side").notNull(), // 'buy' or 'sell'
  amount: doublePrecision("amount").notNull(),
  triggerPrice: doublePrecision("trigger_price"), // Price to trigger the order
  limitPrice: doublePrecision("limit_price"), // Limit price for execution
  stopPrice: doublePrecision("stop_price"), // Stop price (for OCO or stop orders)
  takeProfitPrice: doublePrecision("take_profit_price"), // Take profit price (for OCO)
  trailingPercent: doublePrecision("trailing_percent"), // For trailing stop orders
  trailingActivationPrice: doublePrecision("trailing_activation_price"), // Price at which trailing starts
  status: text("status").notNull().default("pending"), // 'pending', 'triggered', 'filled', 'cancelled', 'expired'
  linkedOrderId: integer("linked_order_id"), // For OCO - linked order that cancels when this fills
  gridId: integer("grid_id"), // For grid orders - which grid this belongs to
  expiresAt: timestamp("expires_at"), // Optional expiration
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
  triggeredAt: timestamp("triggered_at"),
  filledAt: timestamp("filled_at"),
  notes: text("notes"),
  exchangeOrderId: text("exchange_order_id"), // Exchange's order ID once placed
});

// Grid Trading Configurations
export const gridConfigs = pgTable("grid_configs", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").references(() => bots.id).notNull(),
  userId: text("user_id").references(() => users.id),
  symbol: text("symbol").notNull(),
  status: text("status").notNull().default("stopped"), // 'running', 'stopped', 'paused'
  gridType: text("grid_type").notNull().default("arithmetic"), // 'arithmetic' (equal spacing) or 'geometric' (% spacing)
  upperPrice: doublePrecision("upper_price").notNull(), // Upper grid boundary
  lowerPrice: doublePrecision("lower_price").notNull(), // Lower grid boundary
  gridLevels: integer("grid_levels").notNull().default(10), // Number of grid lines
  amountPerGrid: doublePrecision("amount_per_grid").notNull(), // Amount per grid order
  totalInvestment: doublePrecision("total_investment").notNull(), // Total capital allocated
  currentPnL: doublePrecision("current_pnl").default(0),
  totalFilled: integer("total_filled").default(0), // Number of filled grid orders
  lastFilledPrice: doublePrecision("last_filled_price"),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
  startedAt: timestamp("started_at"),
  stoppedAt: timestamp("stopped_at"),
});

// Portfolio Allocations for Rebalancing
export const portfolioAllocations = pgTable("portfolio_allocations", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").references(() => bots.id).notNull(),
  userId: text("user_id").references(() => users.id),
  symbol: text("symbol").notNull(),
  targetPercent: doublePrecision("target_percent").notNull(), // Target % of portfolio
  currentPercent: doublePrecision("current_percent").default(0), // Current actual %
  currentAmount: doublePrecision("current_amount").default(0), // Current holdings
  currentValue: doublePrecision("current_value").default(0), // Current USD value
  rebalanceThreshold: doublePrecision("rebalance_threshold").default(5), // % deviation to trigger rebalance
  isActive: boolean("is_active").default(true),
  lastRebalanced: timestamp("last_rebalanced"),
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
});

// Rebalancing Schedule
export const rebalanceSchedules = pgTable("rebalance_schedules", {
  id: serial("id").primaryKey(),
  botId: integer("bot_id").references(() => bots.id).notNull(),
  userId: text("user_id").references(() => users.id),
  scheduleType: text("schedule_type").notNull().default("manual"), // 'manual', 'daily', 'weekly', 'threshold'
  isEnabled: boolean("is_enabled").default(false),
  lastRun: timestamp("last_run"),
  nextRun: timestamp("next_run"),
  thresholdPercent: doublePrecision("threshold_percent").default(5), // Deviation % for threshold-based
  createdAt: timestamp("created_at").default(sql`CURRENT_TIMESTAMP`).notNull(),
});

export const insertPendingOrderSchema = createInsertSchema(pendingOrders).omit({ id: true, createdAt: true, triggeredAt: true, filledAt: true });
export const insertGridConfigSchema = createInsertSchema(gridConfigs).omit({ id: true, createdAt: true, startedAt: true, stoppedAt: true });
export const insertPortfolioAllocationSchema = createInsertSchema(portfolioAllocations).omit({ id: true, createdAt: true, lastRebalanced: true });
export const insertRebalanceScheduleSchema = createInsertSchema(rebalanceSchedules).omit({ id: true, createdAt: true, lastRun: true, nextRun: true });

export type PendingOrder = typeof pendingOrders.$inferSelect;
export type InsertPendingOrder = z.infer<typeof insertPendingOrderSchema>;
export type GridConfig = typeof gridConfigs.$inferSelect;
export type InsertGridConfig = z.infer<typeof insertGridConfigSchema>;
export type PortfolioAllocation = typeof portfolioAllocations.$inferSelect;
export type InsertPortfolioAllocation = z.infer<typeof insertPortfolioAllocationSchema>;
export type RebalanceSchedule = typeof rebalanceSchedules.$inferSelect;
export type InsertRebalanceSchedule = z.infer<typeof insertRebalanceScheduleSchema>;
