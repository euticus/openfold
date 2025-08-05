# OdinFold++ Visual Assets

## 🎨 **Logo Specifications**

### **Primary Logo**
```
🧬 OdinFold++
```

**Components:**
- **Symbol**: DNA helix emoji (🧬) - represents genetic/protein science
- **Wordmark**: "OdinFold++" in Inter Bold font
- **Plus Signs**: Represent enhancement, evolution, advancement
- **Spacing**: 0.5em between symbol and text

### **Logo Variations**

#### **Full Logo (Horizontal)**
```
🧬 OdinFold++
   Next-generation protein folding
```

#### **Compact Logo**
```
🧬 OF++
```

#### **Text-Only Logo**
```
OdinFold++
```

#### **Symbol-Only Logo**
```
🧬
```

### **Logo Colors**

#### **Primary (Light Backgrounds)**
- Symbol: #2563eb (Primary Blue)
- Text: #1e293b (Dark Gray)
- Tagline: #64748b (Medium Gray)

#### **Inverse (Dark Backgrounds)**
- Symbol: #60a5fa (Light Blue)
- Text: #ffffff (White)
- Tagline: #cbd5e1 (Light Gray)

#### **Monochrome**
- All elements: #1e293b (Dark Gray) or #ffffff (White)

## 🎨 **Color System**

### **Primary Palette**
```css
/* Primary Blue - Technology, Trust, Reliability */
--primary-50:  #eff6ff
--primary-100: #dbeafe
--primary-200: #bfdbfe
--primary-300: #93c5fd
--primary-400: #60a5fa
--primary-500: #3b82f6  /* Main Primary */
--primary-600: #2563eb  /* Logo Primary */
--primary-700: #1d4ed8
--primary-800: #1e40af
--primary-900: #1e3a8a

/* Success Green - Achievement, Accuracy, Growth */
--success-50:  #ecfdf5
--success-100: #d1fae5
--success-200: #a7f3d0
--success-300: #6ee7b7
--success-400: #34d399
--success-500: #10b981  /* Main Success */
--success-600: #059669
--success-700: #047857
--success-800: #065f46
--success-900: #064e3b

/* Warning Amber - Attention, Caution, Energy */
--warning-50:  #fffbeb
--warning-100: #fef3c7
--warning-200: #fde68a
--warning-300: #fcd34d
--warning-400: #fbbf24
--warning-500: #f59e0b  /* Main Warning */
--warning-600: #d97706
--warning-700: #b45309
--warning-800: #92400e
--warning-900: #78350f
```

### **Neutral Palette**
```css
/* Grays - Professional, Clean, Modern */
--gray-50:  #f8fafc
--gray-100: #f1f5f9
--gray-200: #e2e8f0
--gray-300: #cbd5e1
--gray-400: #94a3b8
--gray-500: #64748b
--gray-600: #475569
--gray-700: #334155
--gray-800: #1e293b
--gray-900: #0f172a
```

### **Accent Colors**
```css
/* Error Red */
--error-500: #ef4444
--error-100: #fecaca

/* Info Cyan */
--info-500: #06b6d4
--info-100: #cffafe

/* Purple (WASM/Browser) */
--purple-500: #8b5cf6
--purple-100: #e9d5ff

/* Indigo (Enterprise) */
--indigo-500: #6366f1
--indigo-100: #e0e7ff
```

## 📝 **Typography System**

### **Font Stack**
```css
/* Primary Font Family */
--font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;

/* Monospace Font Family */
--font-mono: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace;
```

### **Type Scale**
```css
/* Display Sizes */
--text-xs:   0.75rem  /* 12px */
--text-sm:   0.875rem /* 14px */
--text-base: 1rem     /* 16px */
--text-lg:   1.125rem /* 18px */
--text-xl:   1.25rem  /* 20px */
--text-2xl:  1.5rem   /* 24px */
--text-3xl:  1.875rem /* 30px */
--text-4xl:  2.25rem  /* 36px */
--text-5xl:  3rem     /* 48px */
--text-6xl:  3.75rem  /* 60px */

/* Font Weights */
--font-light:     300
--font-normal:    400
--font-medium:    500
--font-semibold:  600
--font-bold:      700
--font-extrabold: 800
```

### **Typography Hierarchy**
```css
/* Headings */
h1 { font: 700 3rem/1.1 var(--font-primary); }      /* 48px Bold */
h2 { font: 600 2.25rem/1.2 var(--font-primary); }   /* 36px SemiBold */
h3 { font: 600 1.875rem/1.3 var(--font-primary); }  /* 30px SemiBold */
h4 { font: 500 1.5rem/1.4 var(--font-primary); }    /* 24px Medium */
h5 { font: 500 1.25rem/1.5 var(--font-primary); }   /* 20px Medium */
h6 { font: 500 1.125rem/1.5 var(--font-primary); }  /* 18px Medium */

/* Body Text */
body { font: 400 1rem/1.6 var(--font-primary); }    /* 16px Normal */
small { font: 400 0.875rem/1.5 var(--font-primary); } /* 14px Normal */

/* Code */
code { font: 400 0.875rem/1.4 var(--font-mono); }   /* 14px Mono */
pre { font: 400 0.875rem/1.6 var(--font-mono); }    /* 14px Mono */
```

## 🎯 **Icon System**

### **Core Icons**
```
🧬 DNA/Protein (primary brand symbol)
⚡ Performance/Speed
🚀 Innovation/Launch
🌐 Universal/Global
🔧 Tools/Engineering
📊 Analytics/Metrics
🎯 Precision/Accuracy
🛡️ Security/Reliability
💻 Development/Code
🔬 Research/Science
⚙️ Configuration/Settings
📈 Growth/Improvement
🎨 Design/Creative
🤝 Community/Collaboration
📚 Documentation/Learning
🏆 Achievement/Success
```

### **Icon Guidelines**
- **Style**: Outline icons with 2px stroke weight
- **Size**: 24px standard (16px, 20px, 32px variants)
- **Color**: Inherit from context or use primary blue
- **Library**: Heroicons, Lucide, or custom SVG
- **Accessibility**: Include proper alt text and ARIA labels

## 🖼️ **Image Guidelines**

### **Photography Style**
- **Subject**: Scientists, researchers, laboratories, technology
- **Mood**: Professional, innovative, collaborative, optimistic
- **Color**: Natural tones with blue/green accents
- **Composition**: Clean, uncluttered, well-lit
- **Quality**: High resolution, sharp focus

### **Illustration Style**
- **Type**: Geometric, modern, scientific
- **Color**: Brand palette with gradients
- **Style**: Flat design with subtle shadows
- **Content**: Protein structures, molecular diagrams, data visualizations

### **Diagram Standards**
- **Background**: White or light gray (#f8fafc)
- **Lines**: 2px stroke, primary blue (#2563eb)
- **Text**: Inter font, dark gray (#1e293b)
- **Highlights**: Success green (#10b981) for positive elements
- **Spacing**: Consistent 16px grid system

## 🎨 **UI Component Library**

### **Buttons**
```css
/* Primary Button */
.btn-primary {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  font: 500 1rem var(--font-primary);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Secondary Button */
.btn-secondary {
  background: white;
  color: #2563eb;
  border: 2px solid #2563eb;
  border-radius: 8px;
  padding: 10px 22px;
  font: 500 1rem var(--font-primary);
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background: #2563eb;
  color: white;
}
```

### **Cards**
```css
.card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
  transition: all 0.2s ease;
}

.card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}
```

### **Code Blocks**
```css
.code-block {
  background: #1e293b;
  color: #e2e8f0;
  border-radius: 8px;
  padding: 16px;
  font: 400 0.875rem var(--font-mono);
  overflow-x: auto;
  border: 1px solid #334155;
}

.code-inline {
  background: #f1f5f9;
  color: #1e293b;
  padding: 2px 6px;
  border-radius: 4px;
  font: 400 0.875rem var(--font-mono);
}
```

## 📱 **Responsive Design**

### **Breakpoints**
```css
/* Mobile First Approach */
--breakpoint-sm: 640px   /* Small devices */
--breakpoint-md: 768px   /* Medium devices */
--breakpoint-lg: 1024px  /* Large devices */
--breakpoint-xl: 1280px  /* Extra large devices */
--breakpoint-2xl: 1536px /* 2X large devices */
```

### **Grid System**
```css
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

@media (min-width: 640px) {
  .container { padding: 0 24px; }
}

@media (min-width: 1024px) {
  .container { padding: 0 32px; }
}
```

## 🌐 **Web Assets**

### **Favicon Sizes**
- 16x16px (browser tab)
- 32x32px (browser bookmark)
- 48x48px (desktop shortcut)
- 180x180px (Apple touch icon)
- 192x192px (Android icon)
- 512x512px (PWA icon)

### **Social Media Assets**
- **Twitter Card**: 1200x630px
- **Facebook Share**: 1200x630px
- **LinkedIn Share**: 1200x627px
- **GitHub Social**: 1280x640px

### **Documentation Assets**
- **Banner Image**: 1200x300px
- **Feature Graphics**: 600x400px
- **Diagram Templates**: 800x600px
- **Screenshot Guidelines**: 1440x900px (16:10 ratio)

## 📄 **Brand Asset Files**

### **Logo Files**
```
/assets/logos/
├── odinfold-logo-primary.svg
├── odinfold-logo-inverse.svg
├── odinfold-logo-monochrome.svg
├── odinfold-logo-compact.svg
├── odinfold-symbol-only.svg
└── odinfold-wordmark.svg
```

### **Color Swatches**
```
/assets/colors/
├── primary-palette.ase
├── brand-colors.sketch
└── color-tokens.json
```

### **Typography**
```
/assets/fonts/
├── Inter-Variable.woff2
├── Inter-Regular.woff2
├── Inter-Medium.woff2
├── Inter-SemiBold.woff2
└── Inter-Bold.woff2
```

---

**These visual assets ensure consistent, professional branding across all OdinFold++ touchpoints while maintaining accessibility and usability standards.**
