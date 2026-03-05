// Sidebar Navigation JavaScript

class Sidebar {
    constructor() {
        this.sidebar = document.querySelector('.sidebar');
        this.toggleBtn = document.querySelector('.sidebar-toggle');
        this.mobileMenuBtn = document.querySelector('.mobile-menu-btn');
        this.navLinks = document.querySelectorAll('.nav-link');

        this.init();
    }

    init() {
        // Load saved state
        const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
        if (isCollapsed) {
            this.sidebar?.classList.add('collapsed');
        }

        // Toggle button click
        this.toggleBtn?.addEventListener('click', () => this.toggle());

        // Mobile menu button
        this.mobileMenuBtn?.addEventListener('click', () => this.toggleMobile());

        // Close mobile sidebar when clicking outside
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (!this.sidebar?.contains(e.target) && !this.mobileMenuBtn?.contains(e.target)) {
                    this.sidebar?.classList.remove('open');
                }
            }
        });

        // Set active link based on current page
        this.setActiveLink();

        // Handle keyboard navigation
        this.setupKeyboardNav();
    }

    toggle() {
        this.sidebar?.classList.toggle('collapsed');
        localStorage.setItem('sidebarCollapsed', this.sidebar?.classList.contains('collapsed'));
    }

    toggleMobile() {
        this.sidebar?.classList.toggle('open');
    }

    setActiveLink() {
        const currentPath = window.location.pathname;
        const currentFile = currentPath.split('/').pop() || 'index.html';

        this.navLinks.forEach(link => {
            const href = link.getAttribute('href');
            const linkFile = href.split('/').pop();

            if (linkFile === currentFile ||
                (currentFile === 'index.html' && linkFile === 'index.html') ||
                (currentFile === '' && linkFile === 'index.html')) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    setupKeyboardNav() {
        document.addEventListener('keydown', (e) => {
            // Toggle sidebar with Ctrl/Cmd + B
            if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
                e.preventDefault();
                this.toggle();
            }

            // Close mobile sidebar with Escape
            if (e.key === 'Escape' && window.innerWidth <= 768) {
                this.sidebar?.classList.remove('open');
            }
        });
    }
}

// Initialize sidebar when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Sidebar();
});

// Methods registry for easy management
const MethodsRegistry = {
    methods: [],

    register(method) {
        this.methods.push(method);
    },

    getByCategory(category) {
        return this.methods.filter(m => m.category === category);
    },

    getAll() {
        return this.methods;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Sidebar, MethodsRegistry };
}
