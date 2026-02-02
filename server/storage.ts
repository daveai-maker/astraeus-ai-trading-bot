import { 
  Bot, InsertBot, Trade, InsertTrade, Log, InsertLog, Backtest, InsertBacktest, 
  ForumCategory, InsertForumCategory, ForumTopic, InsertForumTopic, ForumPost, InsertForumPost,
  PendingOrder, InsertPendingOrder, GridConfig, InsertGridConfig, PortfolioAllocation, InsertPortfolioAllocation,
  RebalanceSchedule, InsertRebalanceSchedule,
  bots, trades, logs, backtests, forumCategories, forumTopics, forumPosts,
  pendingOrders, gridConfigs, portfolioAllocations, rebalanceSchedules
} from "@shared/schema";
import { db } from "./db";
import { eq, desc, asc, sql, and } from "drizzle-orm";

export interface IStorage {
  getBot(id: number): Promise<Bot | undefined>;
  getBotByUserId(userId: string): Promise<Bot | undefined>;
  getAllBots(): Promise<Bot[]>;
  getOrCreateBotForUser(userId: string): Promise<Bot>;
  updateBot(id: number, bot: Partial<Bot>): Promise<Bot>;
  getTrades(botId: number): Promise<Trade[]>;
  createTrade(trade: InsertTrade): Promise<Trade>;
  updateTrade(id: number, trade: Partial<Trade>): Promise<Trade | undefined>;
  clearTrades(botId: number): Promise<void>;
  clearPaperTrades(botId: number): Promise<void>;
  getLogs(botId: number): Promise<Log[]>;
  createLog(log: InsertLog): Promise<Log>;
  getBacktests(botId: number): Promise<Backtest[]>;
  createBacktest(backtest: InsertBacktest): Promise<Backtest>;
  // Forum operations
  getForumCategories(): Promise<ForumCategory[]>;
  createForumCategory(category: InsertForumCategory): Promise<ForumCategory>;
  getForumTopics(categoryId?: number): Promise<ForumTopic[]>;
  getForumTopic(id: number): Promise<ForumTopic | undefined>;
  createForumTopic(topic: InsertForumTopic): Promise<ForumTopic>;
  updateForumTopic(id: number, update: Partial<ForumTopic>): Promise<ForumTopic | undefined>;
  getForumPosts(topicId: number): Promise<ForumPost[]>;
  createForumPost(post: InsertForumPost): Promise<ForumPost>;
  upvoteForumTopic(id: number): Promise<ForumTopic | undefined>;
  upvoteForumPost(id: number): Promise<ForumPost | undefined>;
}

export class DatabaseStorage implements IStorage {
  async getBot(id: number): Promise<Bot | undefined> {
    const [bot] = await db.select().from(bots).where(eq(bots.id, id));
    return bot;
  }

  async getBotByUserId(userId: string): Promise<Bot | undefined> {
    const [bot] = await db.select().from(bots).where(eq(bots.userId, userId));
    return bot;
  }

  async getAllBots(): Promise<Bot[]> {
    return db.select().from(bots);
  }

  async getOrCreateBotForUser(userId: string): Promise<Bot> {
    const existing = await this.getBotByUserId(userId);
    if (existing) {
      return existing;
    }
    const [created] = await db.insert(bots).values({
      userId,
      name: "Astraeus AI",
      isRunning: false,
      isLiveMode: false,
      exchange: "coinbase",
      symbol: "BTC/USDT",
      intervalSeconds: 60,
      paperBalance: 10000,
      paperStartingCapital: 10000,
    }).returning();
    return created;
  }

  async updateBot(id: number, botUpdate: Partial<Bot>): Promise<Bot> {
    const existing = await this.getBot(id);
    if (!existing) {
      const [created] = await db.insert(bots).values(botUpdate as any).returning();
      return created;
    }
    const [updated] = await db.update(bots).set(botUpdate).where(eq(bots.id, id)).returning();
    return updated;
  }

  async getTrades(botId: number): Promise<Trade[]> {
    return db.select().from(trades).where(eq(trades.botId, botId)).orderBy(desc(trades.timestamp));
  }

  async createTrade(trade: InsertTrade): Promise<Trade> {
    const [created] = await db.insert(trades).values(trade).returning();
    return created;
  }

  async updateTrade(id: number, tradeUpdate: Partial<Trade>): Promise<Trade | undefined> {
    const [updated] = await db.update(trades).set(tradeUpdate).where(eq(trades.id, id)).returning();
    return updated;
  }

  async clearTrades(botId: number): Promise<void> {
    await db.delete(trades).where(eq(trades.botId, botId));
  }

  async clearPaperTrades(botId: number): Promise<void> {
    const { and } = await import("drizzle-orm");
    await db.delete(trades).where(and(eq(trades.botId, botId), eq(trades.isPaperTrade, true)));
  }

  async getLogs(botId: number): Promise<Log[]> {
    return db.select().from(logs).where(eq(logs.botId, botId)).orderBy(desc(logs.timestamp)).limit(100);
  }

  async createLog(log: InsertLog): Promise<Log> {
    const [created] = await db.insert(logs).values(log).returning();
    return created;
  }

  async getBacktests(botId: number): Promise<Backtest[]> {
    return db.select().from(backtests).where(eq(backtests.botId, botId)).orderBy(desc(backtests.timestamp));
  }

  async createBacktest(backtest: InsertBacktest): Promise<Backtest> {
    const [created] = await db.insert(backtests).values(backtest).returning();
    return created;
  }

  // Forum operations
  async getForumCategories(): Promise<ForumCategory[]> {
    return db.select().from(forumCategories).orderBy(asc(forumCategories.sortOrder));
  }

  async createForumCategory(category: InsertForumCategory): Promise<ForumCategory> {
    const [created] = await db.insert(forumCategories).values(category).returning();
    return created;
  }

  async getForumTopics(categoryId?: number): Promise<ForumTopic[]> {
    if (categoryId) {
      return db.select().from(forumTopics)
        .where(eq(forumTopics.categoryId, categoryId))
        .orderBy(desc(forumTopics.isPinned), desc(forumTopics.createdAt));
    }
    return db.select().from(forumTopics)
      .orderBy(desc(forumTopics.isPinned), desc(forumTopics.createdAt));
  }

  async getForumTopic(id: number): Promise<ForumTopic | undefined> {
    const [topic] = await db.update(forumTopics)
      .set({ viewCount: sql`${forumTopics.viewCount} + 1` })
      .where(eq(forumTopics.id, id))
      .returning();
    return topic;
  }

  async createForumTopic(topic: InsertForumTopic): Promise<ForumTopic> {
    return db.transaction(async (tx) => {
      const [created] = await tx.insert(forumTopics).values(topic).returning();
      await tx.update(forumCategories)
        .set({ topicCount: sql`${forumCategories.topicCount} + 1` })
        .where(eq(forumCategories.id, topic.categoryId));
      return created;
    });
  }

  async updateForumTopic(id: number, update: Partial<ForumTopic>): Promise<ForumTopic | undefined> {
    const [updated] = await db.update(forumTopics).set(update).where(eq(forumTopics.id, id)).returning();
    return updated;
  }

  async getForumPosts(topicId: number): Promise<ForumPost[]> {
    return db.select().from(forumPosts)
      .where(eq(forumPosts.topicId, topicId))
      .orderBy(asc(forumPosts.createdAt));
  }

  async createForumPost(post: InsertForumPost): Promise<ForumPost> {
    return db.transaction(async (tx) => {
      const [created] = await tx.insert(forumPosts).values(post).returning();
      await tx.update(forumTopics)
        .set({ 
          replyCount: sql`${forumTopics.replyCount} + 1`,
          lastReplyAt: sql`CURRENT_TIMESTAMP`,
          updatedAt: sql`CURRENT_TIMESTAMP`
        })
        .where(eq(forumTopics.id, post.topicId));
      return created;
    });
  }

  async upvoteForumTopic(id: number): Promise<ForumTopic | undefined> {
    const [updated] = await db.update(forumTopics)
      .set({ upvotes: sql`${forumTopics.upvotes} + 1` })
      .where(eq(forumTopics.id, id))
      .returning();
    return updated;
  }

  async upvoteForumPost(id: number): Promise<ForumPost | undefined> {
    const [updated] = await db.update(forumPosts)
      .set({ upvotes: sql`${forumPosts.upvotes} + 1` })
      .where(eq(forumPosts.id, id))
      .returning();
    return updated;
  }

  // Pending Orders
  async getPendingOrders(botId: number): Promise<PendingOrder[]> {
    return db.select().from(pendingOrders)
      .where(eq(pendingOrders.botId, botId))
      .orderBy(desc(pendingOrders.createdAt));
  }

  async getPendingOrdersByStatus(botId: number, status: string): Promise<PendingOrder[]> {
    return db.select().from(pendingOrders)
      .where(and(eq(pendingOrders.botId, botId), eq(pendingOrders.status, status)))
      .orderBy(desc(pendingOrders.createdAt));
  }

  async createPendingOrder(order: InsertPendingOrder): Promise<PendingOrder> {
    const [created] = await db.insert(pendingOrders).values(order).returning();
    return created;
  }

  async updatePendingOrder(id: number, update: Partial<PendingOrder>): Promise<PendingOrder | undefined> {
    const [updated] = await db.update(pendingOrders).set(update).where(eq(pendingOrders.id, id)).returning();
    return updated;
  }

  async deletePendingOrder(id: number): Promise<void> {
    await db.delete(pendingOrders).where(eq(pendingOrders.id, id));
  }

  // Grid Configs
  async getGridConfigs(botId: number): Promise<GridConfig[]> {
    return db.select().from(gridConfigs)
      .where(eq(gridConfigs.botId, botId))
      .orderBy(desc(gridConfigs.createdAt));
  }

  async getGridConfig(id: number): Promise<GridConfig | undefined> {
    const [config] = await db.select().from(gridConfigs).where(eq(gridConfigs.id, id));
    return config;
  }

  async createGridConfig(config: InsertGridConfig): Promise<GridConfig> {
    const [created] = await db.insert(gridConfigs).values(config).returning();
    return created;
  }

  async updateGridConfig(id: number, update: Partial<GridConfig>): Promise<GridConfig | undefined> {
    const [updated] = await db.update(gridConfigs).set(update).where(eq(gridConfigs.id, id)).returning();
    return updated;
  }

  async deleteGridConfig(id: number): Promise<void> {
    await db.delete(gridConfigs).where(eq(gridConfigs.id, id));
  }

  // Portfolio Allocations
  async getPortfolioAllocations(botId: number): Promise<PortfolioAllocation[]> {
    return db.select().from(portfolioAllocations)
      .where(eq(portfolioAllocations.botId, botId))
      .orderBy(desc(portfolioAllocations.targetPercent));
  }

  async createPortfolioAllocation(allocation: InsertPortfolioAllocation): Promise<PortfolioAllocation> {
    const [created] = await db.insert(portfolioAllocations).values(allocation).returning();
    return created;
  }

  async updatePortfolioAllocation(id: number, update: Partial<PortfolioAllocation>): Promise<PortfolioAllocation | undefined> {
    const [updated] = await db.update(portfolioAllocations).set(update).where(eq(portfolioAllocations.id, id)).returning();
    return updated;
  }

  async deletePortfolioAllocation(id: number): Promise<void> {
    await db.delete(portfolioAllocations).where(eq(portfolioAllocations.id, id));
  }

  // Rebalance Schedules
  async getRebalanceSchedule(botId: number): Promise<RebalanceSchedule | undefined> {
    const [schedule] = await db.select().from(rebalanceSchedules).where(eq(rebalanceSchedules.botId, botId));
    return schedule;
  }

  async createRebalanceSchedule(schedule: InsertRebalanceSchedule): Promise<RebalanceSchedule> {
    const [created] = await db.insert(rebalanceSchedules).values(schedule).returning();
    return created;
  }

  async updateRebalanceSchedule(id: number, update: Partial<RebalanceSchedule>): Promise<RebalanceSchedule | undefined> {
    const [updated] = await db.update(rebalanceSchedules).set(update).where(eq(rebalanceSchedules.id, id)).returning();
    return updated;
  }
}

export const storage = new DatabaseStorage();
