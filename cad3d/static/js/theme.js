(() => {
  'use strict';
  
  const THEME_KEY = 'ai_theme_pref';
  const TRANSITION_CLASS = 'theme-transitioning';
  
  // Apply theme immediately (before DOM ready) to prevent flash
  const applyTheme = (themeName) => {
    document.documentElement.setAttribute('data-theme', themeName);
    localStorage.setItem(THEME_KEY, themeName);
  };
  
  // Check for saved theme or system preference
  const savedTheme = localStorage.getItem(THEME_KEY);
  if (savedTheme) {
    applyTheme(savedTheme);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    if (currentTheme === 'light') {
      applyTheme('dark');
    }
  }
  
  document.addEventListener('DOMContentLoaded', () => {
    const switcher = document.getElementById('themeSwitcher');
    if (!switcher) return;
    
    // Update active chip on load
    const updateActiveChip = (themeName) => {
      switcher.querySelectorAll('.theme-chip').forEach(chip => {
        chip.classList.toggle('active', chip.getAttribute('data-theme') === themeName);
      });
    };
    
    const currentTheme = document.documentElement.getAttribute('data-theme');
    updateActiveChip(currentTheme);
    
    // Handle theme switching
    switcher.addEventListener('click', e => {
      const chip = e.target.closest('.theme-chip');
      if (!chip) return;
      
      const theme = chip.getAttribute('data-theme');
      if (!theme) return;
      
      // Add transition class for smooth change
      document.body.classList.add(TRANSITION_CLASS);
      
      // Update UI immediately
      updateActiveChip(theme);
      applyTheme(theme);
      
      // Send to server
      const form = new FormData();
      form.append('theme', theme);
      fetch('/api/themes/select', { 
        method: 'POST', 
        body: form,
        credentials: 'same-origin'
      }).then(r => {
        if (!r.ok) console.warn('Theme save failed:', r.status);
      }).catch(err => console.warn('Theme switch request failed:', err));
      
      // Remove transition class after animation
      setTimeout(() => document.body.classList.remove(TRANSITION_CLASS), 400);
    });
    
    // Keyboard navigation (Arrow keys)
    switcher.addEventListener('keydown', e => {
      if (!['ArrowLeft', 'ArrowRight', 'Enter', ' '].includes(e.key)) return;
      
      const chips = Array.from(switcher.querySelectorAll('.theme-chip'));
      const activeIndex = chips.findIndex(c => c.classList.contains('active'));
      
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const prevIndex = (activeIndex - 1 + chips.length) % chips.length;
        chips[prevIndex].focus();
        chips[prevIndex].click();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        const nextIndex = (activeIndex + 1) % chips.length;
        chips[nextIndex].focus();
        chips[nextIndex].click();
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        e.target.click();
      }
    });
    
    // Make chips focusable
    switcher.querySelectorAll('.theme-chip').forEach((chip, i) => {
      chip.setAttribute('tabindex', i === 0 ? '0' : '-1');
      chip.setAttribute('role', 'button');
      chip.setAttribute('aria-label', `تم ${chip.textContent.trim()}`);
    });
    
    // Listen for system theme changes
    if (window.matchMedia) {
      const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
      darkModeQuery.addEventListener('change', e => {
        if (!localStorage.getItem(THEME_KEY)) {
          applyTheme(e.matches ? 'dark' : 'light');
          updateActiveChip(e.matches ? 'dark' : 'light');
        }
      });
    }
  });
})();