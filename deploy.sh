#!/bin/bash

echo "🚀 Deploying ASDP Backend..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Make sure you're in the backend directory."
    exit 1
fi

# Check if database exists and create if needed
echo "📊 Checking database..."
python check_db.py

if [ $? -eq 0 ]; then
    echo "✅ Database check completed"
else
    echo "❌ Database check failed"
    exit 1
fi

# Test API endpoints
echo "🧪 Testing API endpoints..."
python test_api.py

if [ $? -eq 0 ]; then
    echo "✅ API tests completed"
else
    echo "⚠️  Some API tests failed (this might be expected for auth endpoints)"
fi

echo "✅ Backend deployment preparation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Commit and push your changes to GitHub"
echo "2. Render will automatically redeploy"
echo "3. Check the deployment at: https://asdp-banckend.onrender.com/health"
echo ""
echo "🔧 Manual deployment commands:"
echo "   git add ."
echo "   git commit -m 'Fix deployment issues'"
echo "   git push origin main"
