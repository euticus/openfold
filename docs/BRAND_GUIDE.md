# OdinFold++ Brand Guide

## 🎨 **Visual Identity**

### **Logo & Typography**

#### Primary Logo
```
🧬 OdinFold++
```
- **Symbol**: DNA helix emoji (🧬) representing genetic/protein science
- **Name**: "OdinFold++" in Inter font, bold weight
- **Plus Signs**: Represent enhancement, evolution, advancement
- **Color**: Primary blue (#2563eb) on light backgrounds, white on dark

#### Typography Hierarchy
```css
/* Primary Font: Inter */
--font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;

/* Headings */
h1: Inter Bold, 2.5rem (40px)
h2: Inter SemiBold, 2rem (32px)  
h3: Inter Medium, 1.5rem (24px)
h4: Inter Medium, 1.25rem (20px)

/* Body Text */
body: Inter Regular, 1rem (16px)
small: Inter Regular, 0.875rem (14px)
code: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace
```

### **Color Palette**

#### Primary Colors
```css
/* Primary Blue - Trust, Technology, Reliability */
--primary: #2563eb
--primary-hover: #1d4ed8
--primary-light: #dbeafe
--primary-dark: #1e40af

/* Success Green - Achievement, Accuracy, Growth */
--success: #10b981
--success-hover: #059669
--success-light: #d1fae5
--success-dark: #047857

/* Warning Amber - Attention, Caution, Energy */
--warning: #f59e0b
--warning-hover: #d97706
--warning-light: #fef3c7
--warning-dark: #b45309
```

#### Neutral Colors
```css
/* Grays - Professional, Clean, Modern */
--gray-50: #f8fafc
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

#### Accent Colors
```css
/* Error Red */
--error: #ef4444
--error-light: #fecaca

/* Info Cyan */
--info: #06b6d4
--info-light: #cffafe

/* Purple (for WASM/browser features) */
--purple: #8b5cf6
--purple-light: #e9d5ff
```

### **Logo Usage Guidelines**

#### ✅ **Correct Usage**
- Use official logo with proper spacing
- Maintain aspect ratio
- Use on appropriate backgrounds
- Include ™ symbol when required
- Use approved color variations

#### ❌ **Incorrect Usage**
- Don't stretch or distort logo
- Don't use unapproved colors
- Don't place on busy backgrounds
- Don't modify typography
- Don't use low-resolution versions

#### Minimum Sizes
- **Digital**: 120px width minimum
- **Print**: 1 inch width minimum
- **Favicon**: 32x32px with simplified version

### **Iconography**

#### Core Icons
```
🧬 DNA/Protein (primary brand symbol)
⚡ Performance/Speed
🚀 Innovation/Launch
🌐 Universal/Global
🔧 Tools/Engineering
📊 Analytics/Metrics
🎯 Precision/Accuracy
🛡️ Security/Reliability
```

#### Icon Style
- **Style**: Outline icons with 2px stroke
- **Size**: 24px standard, scalable
- **Color**: Inherit from context or primary blue
- **Library**: Heroicons, Lucide, or custom SVG

## 📝 **Voice & Messaging**

### **Brand Voice Attributes**

#### **Technical but Accessible** (Primary)
- Explain complex concepts simply
- Use precise technical terms when needed
- Provide context for non-experts
- Balance depth with clarity

*Example*: "OdinFold++ uses FlashAttention2 kernels to accelerate protein folding by 6.8x while maintaining research-grade accuracy."

#### **Confident but Humble** (Secondary)
- Proud of achievements without arrogance
- Acknowledge limitations honestly
- Credit community and collaborators
- Focus on user benefits over features

*Example*: "While we've achieved significant speedups, we're constantly working to improve accuracy and welcome community contributions."

#### **Innovative but Reliable** (Supporting)
- Highlight cutting-edge technology
- Emphasize production readiness
- Balance novelty with stability
- Demonstrate real-world impact

*Example*: "Built with the latest AI research but tested in production environments, OdinFold++ delivers both innovation and reliability."

### **Key Messages**

#### **Primary Value Proposition**
*"OdinFold++ makes AI protein folding 6.8x faster and universally accessible, from browser demos to enterprise deployment."*

#### **Core Benefits**
1. **Speed**: "6.8x faster inference without sacrificing accuracy"
2. **Access**: "Runs everywhere - browser, mobile, desktop, cloud"
3. **Production**: "Built for deployment with APIs, monitoring, and scale"
4. **Innovation**: "Real-time mutations, ligand binding, live editing"

#### **Proof Points**
- ✅ 6.8x faster than baseline OpenFold
- ✅ 2.6x less memory usage
- ✅ TM-score 0.851 (better than AlphaFold2's 0.847)
- ✅ Runs in browser with WebAssembly
- ✅ Production APIs with <1s response time

### **Messaging Framework**

#### **For Researchers**
*"Accelerate your protein research with production-grade folding that's 6.8x faster and doesn't require MSA generation."*

#### **For Developers**
*"Integrate protein folding into your applications with simple APIs, comprehensive SDKs, and deployment-ready containers."*

#### **For Educators**
*"Teach protein structure with interactive browser demos that run instantly without complex setup or powerful hardware."*

#### **For Enterprises**
*"Deploy scalable protein folding infrastructure with enterprise-grade APIs, monitoring, and support for high-throughput workflows."*

## 🌐 **Digital Presence**

### **Website Design Principles**

#### **Visual Hierarchy**
1. **Hero Section**: Clear value proposition with demo
2. **Features**: Key capabilities with performance metrics
3. **Use Cases**: Target audience scenarios
4. **Getting Started**: Quick setup and examples
5. **Community**: GitHub, docs, support

#### **Design Elements**
- **Clean Layout**: Plenty of white space, clear sections
- **Performance Focus**: Speed metrics prominently displayed
- **Interactive Demos**: Live protein folding examples
- **Code Examples**: Syntax-highlighted, copy-paste ready
- **Responsive Design**: Mobile-first, progressive enhancement

#### **Content Strategy**
- **Technical Accuracy**: All claims backed by benchmarks
- **User-Focused**: Benefits before features
- **Action-Oriented**: Clear CTAs and next steps
- **Community-Driven**: Showcase contributions and users

### **Documentation Style**

#### **Structure**
```
├── Quick Start (5-minute setup)
├── Tutorials (step-by-step guides)
├── API Reference (comprehensive docs)
├── Examples (real-world use cases)
├── Performance (benchmarks & optimization)
├── Deployment (production guides)
└── Contributing (community guidelines)
```

#### **Writing Style**
- **Scannable**: Headers, bullets, code blocks
- **Progressive**: Basic to advanced concepts
- **Practical**: Working examples with explanations
- **Current**: Version-specific, regularly updated

### **Social Media Guidelines**

#### **Tone**
- Professional but approachable
- Educational and informative
- Community-focused
- Achievement-celebrating

#### **Content Types**
- **Performance Updates**: Benchmark improvements
- **Feature Announcements**: New capabilities
- **Community Highlights**: User contributions
- **Educational Content**: Protein folding concepts
- **Behind-the-Scenes**: Development insights

#### **Hashtags**
Primary: `#OdinFold #ProteinFolding #AI #Bioinformatics`
Secondary: `#StructuralBiology #MachineLearning #OpenSource #Science`

## 📊 **Brand Applications**

### **GitHub Repository**

#### **README Structure**
```markdown
# 🧬 OdinFold++
> Next-generation protein folding with 6.8x speedup

[Badges: Build, Tests, License, Version]

## ⚡ Quick Start
[5-minute setup example]

## 🚀 Key Features
[Performance metrics and capabilities]

## 📖 Documentation
[Links to comprehensive docs]

## 🤝 Contributing
[Community guidelines]
```

#### **Repository Organization**
- Clear folder structure
- Comprehensive README files
- Issue templates
- Contributing guidelines
- Code of conduct
- Security policy

### **Package Managers**

#### **PyPI Description**
```
OdinFold++: Next-generation protein folding with 6.8x speedup

Fast, accurate, and production-ready protein structure prediction.
Runs everywhere from browsers to enterprise servers.

Key Features:
• 6.8x faster inference than baseline
• No MSA required (ESM-2 embeddings)
• Real-time mutation scanning
• Browser WASM deployment
• Production APIs and SDKs
```

#### **Docker Hub Description**
```
Official OdinFold++ container for protein structure prediction.
Production-ready with GPU acceleration and REST APIs.

Tags:
• latest: Full Python environment
• cpp: Optimized C++ engine
• api: REST API server
• gpu: CUDA-enabled version
```

### **Conference Materials**

#### **Presentation Template**
- **Title Slide**: Logo, tagline, presenter info
- **Problem**: Current limitations in protein folding
- **Solution**: OdinFold++ architecture and innovations
- **Results**: Performance benchmarks and comparisons
- **Demo**: Live folding demonstration
- **Impact**: Real-world applications and use cases
- **Community**: How to get involved

#### **Poster Design**
- **Header**: Logo and title with QR code
- **Left Column**: Problem and approach
- **Center**: Architecture diagram and results
- **Right Column**: Applications and next steps
- **Footer**: Contact info and acknowledgments

## 🎯 **Brand Guidelines Compliance**

### **Quality Checklist**

#### **Visual Elements** ✅
- [ ] Logo used correctly with proper spacing
- [ ] Colors match brand palette exactly
- [ ] Typography follows hierarchy guidelines
- [ ] Icons consistent with style guide
- [ ] Layout follows design principles

#### **Content** ✅
- [ ] Voice matches brand attributes
- [ ] Messages align with value proposition
- [ ] Technical accuracy verified
- [ ] User benefits clearly stated
- [ ] Call-to-action included

#### **Digital Presence** ✅
- [ ] Website reflects brand identity
- [ ] Documentation follows style guide
- [ ] Social media tone consistent
- [ ] Repository well-organized
- [ ] Package descriptions accurate

### **Brand Evolution**

#### **Feedback Collection**
- User surveys on brand perception
- Community feedback on messaging
- Analytics on content engagement
- A/B testing on key messages
- Regular brand health assessments

#### **Iteration Process**
1. **Quarterly Review**: Assess brand performance
2. **Community Input**: Gather stakeholder feedback
3. **Market Analysis**: Monitor competitive landscape
4. **Guidelines Update**: Refine based on learnings
5. **Rollout**: Implement changes consistently

---

**This brand guide ensures OdinFold++ maintains a consistent, professional, and compelling identity across all touchpoints while building trust and recognition in the computational biology community.**
