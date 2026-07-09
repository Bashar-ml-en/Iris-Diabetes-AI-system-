# Stage 1: Build the React Application
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency configs
COPY package.json ./
RUN npm install

# Copy source and configurations
COPY tsconfig.json vite.config.ts index.html ./
COPY src/ src/

# Build static assets
RUN npm run build

# Stage 2: Serve Assets via Nginx
FROM nginx:alpine

# Copy built static assets from Stage 1
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
