# ── Frontend Dockerfile ────────────────────────────────────────────────────
# Multi-stage: install → build → serve with Next.js standalone

FROM node:20-alpine AS deps
WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --prefer-offline 2>/dev/null || npm install

# ── Build ────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ── Production ───────────────────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy standalone build
# public/ may be empty — create it first so the copy never fails
RUN mkdir -p ./public
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
