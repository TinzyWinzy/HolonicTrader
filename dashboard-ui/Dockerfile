# Stage 1: Build the React application
FROM node:20-alpine as build

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build the application
# We set VITE_API_URL to empty string so it defaults to relative paths (handled by Nginx proxy)
# However, the code uses "|| 'http://localhost:5000'".
# To force it to use relative (i.e., same origin), we can set it to a special value or change the code logic.
# But effectively, if we want it to use the proxy, we want it to be empty or "/"
ENV VITE_API_URL=""
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
