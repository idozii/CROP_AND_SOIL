# Deployment Instructions for Vercel

## Files Created/Modified for Deployment:
1. ✅ vercel.json - Vercel configuration
2. ✅ index.py - Entry point for Vercel
3. ✅ requirements.txt - Minimal dependencies (original backed up as requirements-full.txt)
4. ✅ .vercelignore - Exclude unnecessary files
5. ✅ runtime.txt - Python version specification
6. ✅ .env - Environment variables
7. ✅ Updated app.py with proper file paths

## Steps to Deploy:

### 1. Install Vercel CLI (if not already installed)
```bash
npm install -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy from your project directory
```bash
cd /home/idozii/Desktop/CROP_AND_SOIL
vercel --prod
```

### 4. Alternative: Deploy via GitHub
- Push your code to GitHub
- Connect your GitHub repository to Vercel dashboard
- Vercel will auto-deploy

## Common Issues and Solutions:

### If you still get 404:
1. **Check build logs** in Vercel dashboard for specific errors
2. **Model file size**: Your models (23MB) might be too large
   - Consider model compression
   - Use external storage (AWS S3, Google Cloud) for models
   
3. **Missing dependencies**: If you get import errors, add missing packages to requirements.txt

### If deployment fails due to size limits:
1. Consider removing some model files temporarily
2. Use model compression techniques
3. Load models from external URLs

### Environment Variables:
In Vercel dashboard, set these if needed:
- `SECRET_KEY`: for Flask sessions
- `FLASK_ENV`: production

## Testing Your Deployment:
1. Visit the Vercel URL after deployment
2. Test user registration/login
3. Test the prediction functionality
4. Check that static files (CSS, images) load correctly

## Backup Information:
- Original requirements.txt saved as requirements-full.txt
- All model files are preserved
- Database will be recreated on first deployment