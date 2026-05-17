# Setup & Deployment Guide

## Local Development Setup

### Quick Start (5 minutes)

1. **Navigate to project directory**
   ```bash
   cd "App For FY"
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open in browser**
   - Visit `http://localhost:3000`
   - Backend should be running at `http://localhost:8000`

### Troubleshooting Setup

**Issue: Port 3000 already in use**
```bash
npm run dev -- --port 3001
```

**Issue: Backend connection errors**
- Ensure FastAPI backend is running: `python -m uvicorn main:app --reload --port 8000`
- Check CORS is enabled in backend
- Verify no firewall blocks `localhost:8000`

**Issue: Dependencies won't install**
```bash
# Clear npm cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

## Production Build

### Build for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` folder ready for deployment.

### Preview Production Build

```bash
npm run preview
```

### Deployment Options

#### Option 1: Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

#### Option 2: Netlify
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

#### Option 3: Docker
Create a `Dockerfile` in the project root:

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Create `nginx.conf`:
```nginx
server {
    listen 80;
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
}
```

Build and run:
```bash
docker build -t ai-detector-frontend .
docker run -p 80:80 ai-detector-frontend
```

#### Option 4: Traditional Web Server (Apache/Nginx)

1. Build: `npm run build`
2. Copy `dist/` contents to web server root
3. Configure server to route all non-file requests to `index.html` (SPA routing)

## Environment Configuration

### Local Development
Create a `.env.local` file:
```
VITE_API_BASE_URL=http://localhost:8000
```

### Production
Create a `.env.production` file:
```
VITE_API_BASE_URL=https://your-api-domain.com
```

Update `src/api/detector.js` to use:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
```

## API Backend Requirements

The frontend requires a FastAPI backend with these endpoints:

### Required Endpoints
- `POST /detect` - Text detection
- `POST /detect/sentences` - Sentence-level analysis
- `POST /detect/file` - Single file detection
- `POST /detect/batch` - Batch file detection
- `POST /explain` - SHAP explainability
- `POST /report` - PDF report generation
- `GET /history` - Submission history
- `GET /stats` - Dashboard statistics
- `GET /health` - Health check

### CORS Configuration
Backend must enable CORS for `http://localhost:3000` (dev) and your production domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Optimization

### Built-in Optimizations
- Lazy component loading with React Router
- Recharts uses responsive containers
- Images optimized through Vite
- CSS tree-shaking with TailwindCSS
- Production bundle optimized

### Additional Tips
- Use CDN for static assets
- Enable gzip compression on server
- Implement caching headers
- Use lighthouse for performance audits

## Monitoring & Debugging

### React DevTools
Install React DevTools browser extension for component inspection.

### Network Debugging
1. Open DevTools (F12)
2. Go to Network tab
3. Monitor API calls to ensure proper backend communication

### Console Errors
Check browser console for errors related to:
- API connection issues
- Missing dependencies
- Route errors

## Scaling Considerations

### For Larger Deployments
1. **Use a real backend** - Don't rely on localhost
2. **Implement rate limiting** - Prevent abuse of API
3. **Add authentication** - Secure user sessions if needed
4. **Database backups** - Ensure API maintains data persistence
5. **Load balancing** - Distribute API requests across servers
6. **CDN integration** - Serve static assets from edge locations

## Security Checklist

- [ ] Remove debug console logs in production
- [ ] Set secure CORS headers
- [ ] Use HTTPS in production
- [ ] Validate all user inputs
- [ ] Keep dependencies updated
- [ ] Hide API keys if any
- [ ] Set secure cookie policies
- [ ] Implement rate limiting

## Maintenance

### Regular Updates
```bash
npm update
npm outdated  # Check for updates
npm audit     # Check for vulnerabilities
npm audit fix
```

### Clearing Cache
```bash
npm cache clean --force
```

### Dependencies Health
```bash
npm ls  # List all dependencies
```

## Support & Resources

- Vite Docs: https://vitejs.dev
- React Docs: https://react.dev
- TailwindCSS: https://tailwindcss.com
- Recharts: https://recharts.org
- React Router: https://reactrouter.com

## Deployment Checklist

Before deploying to production:

- [ ] Build completes without errors: `npm run build`
- [ ] No TypeScript errors
- [ ] No console errors in production build
- [ ] Backend API is accessible
- [ ] CORS is configured properly
- [ ] Environment variables are set
- [ ] All routes work correctly
- [ ] Responsive design tested on mobile
- [ ] Performance meets requirements
- [ ] Security best practices applied
