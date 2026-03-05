'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { methods } from '@/lib/methods-registry';
import { useTheme } from '@/hooks/useTheme';
import s from './Sidebar.module.css';

interface NavItem {
  href: string;
  abbrev: string;
  iconClass: string;
  label: string;
  sublabel: string;
  tooltip: string;
}

function buildSortedItems(
  category: string,
  labelFn: (m: (typeof methods)[number]) => string,
  tooltipFn?: (m: (typeof methods)[number]) => string,
): NavItem[] {
  return methods
    .filter((m) => m.category === category)
    .sort((a, b) => a.year - b.year)
    .map((m) => ({
      href: `/methods/${m.slug}`,
      abbrev: m.abbrev,
      iconClass: m.iconClass,
      label: labelFn(m),
      sublabel: `${m.authors}, ${m.year}`,
      tooltip: tooltipFn ? tooltipFn(m) : m.title,
    }));
}

const navSections: { title: string; items: NavItem[] }[] = [
  {
    title: 'Overview',
    items: [
      {
        href: '/',
        abbrev: 'H',
        iconClass: 'other',
        label: 'Home',
        sublabel: 'All methods overview',
        tooltip: 'Home',
      },
      {
        href: '/lineage',
        abbrev: 'L',
        iconClass: 'other',
        label: 'Paper Lineage',
        sublabel: 'Dependency graph',
        tooltip: 'Lineage',
      },
    ],
  },
  {
    title: 'Standard RL',
    items: buildSortedItems(
      'standard-rl',
      (m) => m.slug.toUpperCase() === 'RLHF' ? 'RLHF' : m.title.split('(')[0].trim().length > 20 ? m.slug.toUpperCase() : m.title.split('(')[0].trim(),
      (m) => m.slug.toUpperCase(),
    ),
  },
  {
    title: 'World Models',
    items: buildSortedItems('world-models', (m) => m.title),
  },
  {
    title: 'Robotics Concepts',
    items: buildSortedItems('robotics', (m) => m.title),
  },
];

function iconClassName(iconClass: string): string {
  const map: Record<string, string> = {
    rl: s.iconRl,
    wm: s.iconWm,
    other: s.iconOther,
    robotics: s.iconRobotics,
  };
  return map[iconClass] ?? s.iconOther;
}

export function Sidebar() {
  const pathname = usePathname();
  const { theme, toggleTheme } = useTheme();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Load saved collapse state
  useEffect(() => {
    const saved = localStorage.getItem('sidebarCollapsed');
    if (saved === 'true') setCollapsed(true);
  }, []);

  const toggle = useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem('sidebarCollapsed', String(next));
      return next;
    });
  }, []);

  // Keyboard: Ctrl+B to toggle, Escape to close mobile
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        toggle();
      }
      if (e.key === 'Escape' && window.innerWidth <= 768) {
        setMobileOpen(false);
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [toggle]);

  // Close mobile sidebar on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (window.innerWidth <= 768) {
        const sidebar = document.querySelector(`.${s.sidebar}`);
        const btn = document.querySelector(`.${s.mobileMenuBtn}`);
        if (
          sidebar &&
          !sidebar.contains(e.target as Node) &&
          btn &&
          !btn.contains(e.target as Node)
        ) {
          setMobileOpen(false);
        }
      }
    }
    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, []);

  // Update main-content margin when collapsed state changes
  useEffect(() => {
    const main = document.querySelector('.main-content') as HTMLElement | null;
    if (main) {
      main.style.marginLeft = collapsed
        ? 'var(--sidebar-collapsed-width)'
        : 'var(--sidebar-width)';
    }
  }, [collapsed]);

  const sidebarClass = [
    s.sidebar,
    collapsed ? s.collapsed : '',
    mobileOpen ? s.open : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <>
      {/* Mobile menu button */}
      <button
        className={s.mobileMenuBtn}
        aria-label="Toggle menu"
        onClick={() => setMobileOpen((o) => !o)}
      >
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <nav className={sidebarClass}>
        {/* Toggle button */}
        <button className={s.toggle} aria-label="Toggle sidebar" onClick={toggle}>
          <svg
            className={s.toggleIcon}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>

        {/* Header */}
        <div className={s.header}>
          <div className={s.logo}>ML</div>
          <div className={s.title}>
            <h1 className={s.titleH1}>ML Methods</h1>
            <span className={s.titleSub}>Interactive Visualizations</span>
          </div>
        </div>

        {/* Nav */}
        <div className={s.nav}>
          {navSections.map((section) => (
            <div className={s.section} key={section.title}>
              <div className={s.sectionTitle}>{section.title}</div>
              <ul className={s.items}>
                {section.items.map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <li className={s.item} key={item.href}>
                      <Link
                        href={item.href}
                        className={`${s.link} ${isActive ? s.active : ''}`}
                        data-tooltip={item.tooltip}
                        onClick={() => setMobileOpen(false)}
                      >
                        <div className={`${s.icon} ${iconClassName(item.iconClass)}`}>
                          {item.abbrev}
                        </div>
                        <div className={s.text}>
                          <span className={s.textPrimary}>{item.label}</span>
                          <span className={s.textSecondary}>{item.sublabel}</span>
                        </div>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className={s.footer}>
          <button
            className={s.themeToggle}
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
            <span className={s.themeLabel}>
              {theme === 'dark' ? 'Light' : 'Dark'}
            </span>
          </button>
          <span className={s.footerText}>
            <kbd className={s.kbd}>Ctrl+B</kbd> toggle sidebar
          </span>
        </div>
      </nav>
    </>
  );
}
