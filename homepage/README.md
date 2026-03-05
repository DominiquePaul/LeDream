# Dream Machines - AI-Powered Robotics Website

<!-- Part of the dreammachines monorepo -->

Professional website for Dream Machines, a stealth robotics startup founded by ML researchers from ETH Zurich, targeting German small and medium-sized manufacturers (Mittelstand) and early-stage investors.

## 🚀 Overview

This is a single-page, bilingual (EN/DE) website built with Next.js 15 that serves as a digital business card and pitch deck. The site emphasizes professional credibility, European engineering values, and academic research background to build trust with risk-averse decision makers in German manufacturing.

## ✨ Key Features

- **Bilingual Support** - Complete EN/DE localization with localStorage persistence
- **Professional Design** - Clean, understated European industrial aesthetic
- **Academic Credibility** - Highlighting ETH Zurich and NeurIPS research background
- **German Legal Compliance** - Full Impressum page with required legal information
- **Responsive Design** - Mobile-first approach optimized for 360px-1440px screens
- **SEO Optimized** - Complete metadata, OpenGraph, and structured data
- **Analytics Ready** - Vercel Analytics integration for visitor tracking

## 🛠️ Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Font**: IBM Plex Sans (multiple weights)
- **Analytics**: Vercel Analytics
- **Deployment**: Vercel (optimized for static export)

## 📁 Project Structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout with metadata and font configuration
│   ├── page.tsx           # Main landing page with all sections
│   ├── globals.css        # Global styles and design system
│   ├── imprint/
│   │   └── page.tsx       # German legal imprint page
│   └── favicon.ico
├── components/
│   ├── LanguageSwitch.tsx # Bilingual toggle with persistence
│   ├── Container.tsx      # Responsive container wrapper
│   ├── Section.tsx        # Consistent section spacing
│   └── ComparisonTable.tsx # Traditional vs AI robotics comparison
└── public/
    ├── robots.txt
    ├── sitemap.xml
    └── og-image-placeholder.md
```

## 🎨 Design System

### Figma Design Tokens

**Colors:**
| Token           | Hex       | Opacity | Usage                        |
|-----------------|-----------|---------|------------------------------|
| Navy (primary)  | `#00288E` | 100%    | Text, headings, land on globe |
| Background      | `#F5F7FF` | 100%    | Page background              |
| Navy accent     | `#00288E` | 10%     | Borders, subtle UI elements  |

**Noise Texture (Figma):**
| Property  | Value     |
|-----------|-----------|
| Type      | Mono      |
| Noise size| 0.4       |
| Density   | 100%      |
| Color     | `#000000` |
| Opacity   | 15%       |

Implemented as an SVG `feTurbulence` filter overlaid on the background via `::before` pseudo-element.

**Typography:**
- Headlines: Season Collection (serif display)
- Body / UI: Hasklig (monospace)
- ASCII art: DM Mono (monospace, Google Fonts)

**Layout:**
- Hero container: max-w-4xl
- Content container: max-w-2xl
- Mobile-first responsive breakpoints

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm/yarn/pnpm

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3001](http://localhost:3001) (or available port) in your browser.

### Build

```bash
npm run build
```

### Production

```bash
npm start
```

## 📄 Content Structure

The website follows this exact section order per the original brief:

1. **Hero** - Value proposition and company tagline
2. **Problem** - European manufacturing challenges
3. **What Our Robots Can Do** - Comparison table + example tasks
4. **Approach** - Technical differentiation
5. **About** - Company mission + founder credibility
6. **Contact** - Status update with email CTA
7. **Footer** - Minimal professional footer with imprint link

## 🌍 Internationalization

Simple locale switching without external i18n libraries:
- Content stored in structured objects within components
- Language preference persisted in localStorage
- Dynamic `<html lang>` attribute updates
- Bilingual footer and legal compliance text

## 📊 SEO & Analytics

**Metadata:**
- Complete OpenGraph and Twitter Card tags
- Structured data for business information
- Multilingual alternate URLs
- Optimized meta descriptions for German B2B audience

**Analytics:**
- Vercel Analytics for privacy-friendly tracking
- GDPR-compliant data collection
- Real-time visitor insights and geographic data

## 🏛️ Legal Compliance

**German Impressum:**
- Complete legal entity information (§5 TMG)
- Content responsibility attribution (§55 Abs. 2 RStV)
- Liability disclaimers and copyright notices
- Privacy policy and data protection statements
- Professional legal language in German/English

## 🎯 Target Audience

**Primary:** Decision makers at German SME manufacturers
- Pragmatic, risk-averse approach
- Values professionalism and clear communication
- Needs credibility signals and European identity

**Secondary:** Early-stage investors
- Evaluating founder credibility and market focus
- Assessing technical competence and business viability

## 🚢 Deployment

Optimized for Vercel deployment:

```bash
vercel --prod
```

**Domain:** Configure custom domain `dream-machines.eu` in Vercel dashboard

**Performance Targets:**
- Lighthouse Performance ≥ 95
- Lighthouse Accessibility ≥ 95
- Lighthouse Best Practices ≥ 95
- Lighthouse SEO ≥ 95

## 📝 Content Management

Email contact: `team@dream-machines.eu`
Legal entity: Dominique Paul, Cologne, Germany

**Key messaging:**
- Academic research credibility (ETH Zurich, NeurIPS)
- European manufacturing focus
- AI-powered flexibility vs traditional automation
- Practical, reliable solutions for real-world variation

## 🔄 Future Enhancements

- [ ] Add ETH Zurich and NeurIPS logos to founder section
- [ ] Create proper og-image.png (1200x630px)
- [ ] Add "Made in Europe" trust badges
- [ ] Implement cookie consent for EU compliance
- [ ] Add case studies/pilot project results
- [ ] Integration with CRM for lead tracking

## 📄 License

© 2025 Dream Machines. Built in Europe.
