"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useResearch } from "./Shell";
import { CATEGORIES } from "@/lib/types";

export default function ResearchSidebar() {
  const pathname = usePathname();
  const { papers, tags, signOut } = useResearch();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("rosSidebarCollapsed");
    if (saved === "true") setCollapsed(true);
  }, []);

  const toggle = useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem("rosSidebarCollapsed", String(next));
      return next;
    });
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "b") {
        e.preventDefault();
        toggle();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [toggle]);

  useEffect(() => {
    const main = document.querySelector(".main-content") as HTMLElement | null;
    if (main) {
      main.style.marginLeft = collapsed
        ? "var(--sidebar-collapsed-width)"
        : "var(--sidebar-width)";
    }
  }, [collapsed]);

  const navItems = [
    { href: "/", label: "Collection", icon: "C", sublabel: `${papers.length} papers` },
    { href: "/lineage", label: "Lineage Graph", icon: "L", sublabel: "Paper dependencies" },
    { href: "/discover", label: "Discovery", icon: "D", sublabel: "New papers" },
    { href: "/authors", label: "Authors", icon: "A", sublabel: "Author network" },
    { href: "/chat", label: "Chat", icon: "?", sublabel: "Ask Claude" },
    { href: "/settings", label: "Settings", icon: "S", sublabel: "Sync & config" },
  ];

  // Category stats
  const catCounts = papers.reduce<Record<string, number>>((acc, p) => {
    acc[p.category] = (acc[p.category] || 0) + 1;
    return acc;
  }, {});

  const sidebarClass = [
    "ros-sidebar",
    collapsed ? "ros-sidebar--collapsed" : "",
    mobileOpen ? "ros-sidebar--open" : "",
  ].filter(Boolean).join(" ");

  return (
    <>
      <button className="ros-mobile-menu" onClick={() => setMobileOpen((o) => !o)}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>

      <nav className={sidebarClass}>
        <button className="ros-sidebar__toggle" onClick={toggle}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>

        {/* Header */}
        <div className="ros-sidebar__header">
          <div className="ros-sidebar__logo">R</div>
          <div className="ros-sidebar__title">
            <h1>Research OS</h1>
            <span>Dream Machines</span>
          </div>
        </div>

        {/* Navigation */}
        <div className="ros-sidebar__nav">
          <div className="ros-sidebar__section">
            <div className="ros-sidebar__section-title">Overview</div>
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`ros-sidebar__link ${pathname === item.href ? "ros-sidebar__link--active" : ""}`}
                onClick={() => setMobileOpen(false)}
              >
                <div className="ros-sidebar__icon">{item.icon}</div>
                <div className="ros-sidebar__text">
                  <span className="ros-sidebar__text-primary">{item.label}</span>
                  <span className="ros-sidebar__text-secondary">{item.sublabel}</span>
                </div>
              </Link>
            ))}
          </div>

          {/* Categories */}
          <div className="ros-sidebar__section">
            <div className="ros-sidebar__section-title">Categories</div>
            {Object.entries(CATEGORIES).map(([key, cat]) => (
              <Link
                key={key}
                href={`/?category=${key}`}
                className={`ros-sidebar__link ${pathname === "/" && typeof window !== "undefined" && new URLSearchParams(window.location.search).get("category") === key ? "ros-sidebar__link--active" : ""}`}
                onClick={() => setMobileOpen(false)}
              >
                <div className="ros-sidebar__icon" style={{ borderColor: cat.color, color: cat.color }}>
                  {key[0].toUpperCase()}
                </div>
                <div className="ros-sidebar__text">
                  <span className="ros-sidebar__text-primary">{cat.label}</span>
                  <span className="ros-sidebar__text-secondary">{catCounts[key] || 0} papers</span>
                </div>
              </Link>
            ))}
          </div>

          {/* Tags */}
          {tags.length > 0 && (
            <div className="ros-sidebar__section">
              <div className="ros-sidebar__section-title">
                Tags
                <Link href="/tags" className="ros-sidebar__section-action" onClick={() => setMobileOpen(false)}>
                  Manage
                </Link>
              </div>
              {tags.map((tag) => (
                <Link
                  key={tag.id}
                  href={`/capsules/${tag.id}`}
                  className={`ros-sidebar__link ${pathname === `/capsules/${tag.id}` ? "ros-sidebar__link--active" : ""}`}
                  onClick={() => setMobileOpen(false)}
                >
                  <div className="ros-sidebar__icon ros-sidebar__icon--tag" style={{ background: tag.color + "22", color: tag.color, borderColor: tag.color + "44" }}>
                    #
                  </div>
                  <div className="ros-sidebar__text">
                    <span className="ros-sidebar__text-primary">{tag.name}</span>
                  </div>
                </Link>
              ))}
            </div>
          )}

        </div>

        {/* Footer */}
        <div className="ros-sidebar__footer">
          <button className="ros-sidebar__signout" onClick={signOut}>
            Sign Out
          </button>
          <span className="ros-sidebar__kbd"><kbd>Ctrl+B</kbd> toggle</span>
        </div>
      </nav>
    </>
  );
}
