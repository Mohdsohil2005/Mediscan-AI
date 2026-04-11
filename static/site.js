document.addEventListener("DOMContentLoaded", () => {
    const langToggle = document.getElementById("langToggle");
    const langText = document.getElementById("langText");
    const siteHeader = document.querySelector(".site-header");
    const navToggle = document.querySelector(".nav-toggle");
    const siteNav = document.getElementById("siteNav");
    const translations = window.pageTranslations || {};
    const initialLang = document.body.dataset.initialLang;
    let lang = initialLang || localStorage.getItem("lang") || "en";
    if (initialLang) {
        localStorage.setItem("lang", initialLang);
    }

    const setToggleState = () => {
        if (!langToggle || !langText) {
            return;
        }
        langToggle.checked = lang === "hi";
        langText.textContent = lang.toUpperCase();
    };

    const applyCommonTranslations = () => {
        document.documentElement.lang = lang;

        document.querySelectorAll("[data-i18n]").forEach((node) => {
            const key = node.dataset.i18n;
            const value = translations[lang] && translations[lang][key];
            if (value) {
                node.textContent = value;
            }
        });

        document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
            const key = node.dataset.i18nPlaceholder;
            const value = translations[lang] && translations[lang][key];
            if (value) {
                node.setAttribute("placeholder", value);
            }
        });

        document.querySelectorAll("[data-lang-link='true']").forEach((node) => {
            const href = node.getAttribute("href");
            if (!href) {
                return;
            }
            const url = new URL(href, window.location.origin);
            url.searchParams.set("lang", lang);
            node.setAttribute("href", `${url.pathname}${url.search}`);
        });

        document.title = document.body.dataset.title || document.title;
    };

    const applyLanguage = () => {
        setToggleState();
        applyCommonTranslations();
        if (typeof window.applyPageLanguage === "function") {
            window.applyPageLanguage(lang);
        }
    };

    if (langToggle) {
        langToggle.addEventListener("change", () => {
            lang = langToggle.checked ? "hi" : "en";
            localStorage.setItem("lang", lang);
            applyLanguage();
        });
    }

    if (navToggle && siteHeader && siteNav) {
        const setNavState = (expanded) => {
            siteHeader.classList.toggle("nav-open", expanded);
            navToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
        };

        navToggle.addEventListener("click", () => {
            const expanded = navToggle.getAttribute("aria-expanded") === "true";
            setNavState(!expanded);
        });

        siteNav.querySelectorAll("a").forEach((link) => {
            link.addEventListener("click", () => {
                if (window.innerWidth <= 640) {
                    setNavState(false);
                }
            });
        });

        window.addEventListener("resize", () => {
            if (window.innerWidth > 640) {
                setNavState(false);
            }
        });
    }

    const revealNodes = document.querySelectorAll(".reveal");
    if (revealNodes.length) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.16 });

        revealNodes.forEach((node) => observer.observe(node));
    }

    applyLanguage();
});
