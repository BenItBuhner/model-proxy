/**
 * Model-Proxy Configuration Interface - Modern JavaScript
 */

// Application State
const state = {
    apiKey: localStorage.getItem('modelProxyApiKey') || '',
    currentPage: 'dashboard',
    providers: [],
    models: [],
    templates: {},
    isLoading: false,
    importData: null,
    editingModel: null,
    editingProvider: null,
    theme: localStorage.getItem('modelProxyTheme') || 'system'
};

// DOM Elements Cache
const elements = {};

// Pre-configured provider list for icons/names
const PROVIDER_INFO = {
    'openai': { icon: 'fa-brain', color: '#10a37f', name: 'OpenAI' },
    'anthropic': { icon: 'fa-sparkles', color: '#d97757', name: 'Anthropic' },
    'gemini': { icon: 'fa-gem', color: '#4285f4', name: 'Google Gemini' },
    'azure': { icon: 'fa-microsoft', color: '#0089d6', name: 'Azure OpenAI' },
    'groq': { icon: 'fa-bolt', color: '#f97316', name: 'Groq' },
    'cerebras': { icon: 'fa-microchip', color: '#7c3aed', name: 'Cerebras' },
    'mistral': { icon: 'fa-wind', color: '#fd714b', name: 'Mistral' },
    'openrouter': { icon: 'fa-route', color: '#6366f1', name: 'OpenRouter' }
};

// Theme configuration
const THEME_CONFIG = {
    system: { icon: 'fa-adjust', label: 'System' },
    light: { icon: 'fa-sun', label: 'Light' },
    dark: { icon: 'fa-moon', label: 'Dark' }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Apply saved theme first
    applyTheme(state.theme);
    
    cacheElements();
    bindEvents();
    
    // Check for stored auth
    if (state.apiKey) {
        verifyAndLoad();
    }
});

// Apply theme to document
function applyTheme(theme) {
    const html = document.documentElement;
    
    if (theme === 'system') {
        // Remove data-theme attribute to let system preference take over
        html.removeAttribute('data-theme');
    } else {
        // Set explicit theme
        html.setAttribute('data-theme', theme);
    }
    
    // Update button if elements are cached
    if (elements.themeIcon && elements.themeLabel) {
        updateThemeButton(theme);
    }
}

// Update theme button appearance
function updateThemeButton(theme) {
    const config = THEME_CONFIG[theme];
    elements.themeIcon.className = `fas ${config.icon}`;
    elements.themeLabel.textContent = config.label;
}

// Cycle through themes: System -> Light -> Dark -> System
function cycleTheme() {
    const themes = ['system', 'light', 'dark'];
    const currentIndex = themes.indexOf(state.theme);
    const nextIndex = (currentIndex + 1) % themes.length;
    const newTheme = themes[nextIndex];
    
    state.theme = newTheme;
    localStorage.setItem('modelProxyTheme', newTheme);
    applyTheme(newTheme);
    
    showToast(`Theme: ${THEME_CONFIG[newTheme].label}`, 'info');
}

// Cache all DOM elements
function cacheElements() {
    // Auth screen
    elements.authScreen = document.getElementById('auth-screen');
    elements.apiKeyInput = document.getElementById('api-key-input');
    elements.authBtn = document.getElementById('auth-btn');
    elements.authError = document.getElementById('auth-error');
    
    // App
    elements.app = document.getElementById('app');
    elements.logoutBtn = document.getElementById('logout-btn');
    
    // Navigation
    elements.navItems = document.querySelectorAll('.nav-item');
    elements.pageTitle = document.getElementById('page-title');
    elements.refreshBtn = document.getElementById('refresh-btn');
    
    // Theme toggle
    elements.themeToggleBtn = document.getElementById('theme-toggle-btn');
    elements.themeIcon = document.getElementById('theme-icon');
    elements.themeLabel = document.getElementById('theme-label');
    
    // Pages
    elements.pages = document.querySelectorAll('.page');
    
    // Dashboard
    elements.dashProviderCount = document.getElementById('dash-provider-count');
    elements.dashEnabledProviders = document.getElementById('dash-enabled-providers');
    elements.dashModelCount = document.getElementById('dash-model-count');
    elements.dashKeysCount = document.getElementById('dash-keys-count');
    elements.dashKeysProviders = document.getElementById('dash-keys-providers');
    elements.dashConfigDir = document.getElementById('dash-config-dir');
    elements.dashLastUpdated = document.getElementById('dash-last-updated');
    
    // Providers page
    elements.preconfiguredProviders = document.getElementById('preconfigured-providers');
    elements.customProvidersList = document.getElementById('custom-providers-list');
    elements.preconfiguredCount = document.getElementById('preconfigured-count');
    elements.customCount = document.getElementById('custom-count');
    elements.addProviderBtn = document.getElementById('add-provider-btn');
    
    // Models page
    elements.modelsList = document.getElementById('models-list');
    elements.addModelBtn = document.getElementById('add-model-btn');
    
    // API Keys page
    elements.apiKeysContainer = document.getElementById('api-keys-container');
    
    // Import/Export page
    elements.exportBtn = document.getElementById('export-btn');
    elements.exportIncludeKeys = document.getElementById('export-include-keys');
    elements.exportPreview = document.getElementById('export-preview');
    elements.importFile = document.getElementById('import-file');
    elements.importFileName = document.getElementById('import-file-name');
    elements.importMerge = document.getElementById('import-merge');
    elements.importBtn = document.getElementById('import-btn');
    elements.importPreview = document.getElementById('import-preview');
    
    // Validation page
    elements.runValidationBtn = document.getElementById('run-validation-btn');
    elements.validationResults = document.getElementById('validation-results');
    
    // Provider Modal
    elements.providerModal = document.getElementById('provider-modal');
    elements.providerName = document.getElementById('provider-name');
    elements.providerDisplayName = document.getElementById('provider-display-name');
    elements.providerBaseUrl = document.getElementById('provider-base-url');
    elements.providerFormat = document.getElementById('provider-format');
    elements.saveProviderBtn = document.getElementById('save-provider-btn');
    elements.providerFormError = document.getElementById('provider-form-error');
    
    // Model Modal
    elements.modelModal = document.getElementById('model-modal');
    elements.modelModalTitle = document.getElementById('model-modal-title');
    elements.modelLogicalName = document.getElementById('model-logical-name');
    elements.modelTimeout = document.getElementById('model-timeout');
    elements.modelCooldown = document.getElementById('model-cooldown');
    elements.modelRoutingsList = document.getElementById('model-routings-list');
    elements.addRoutingBtn = document.getElementById('add-routing-btn');
    elements.saveModelBtn = document.getElementById('save-model-btn');
    elements.modelFormError = document.getElementById('model-form-error');
    
    // Toast container
    elements.toastContainer = document.getElementById('toast-container');
    
    // Update theme button to match current theme
    updateThemeButton(state.theme);
}

// Bind all event listeners
function bindEvents() {
    // Auth
    elements.authBtn.addEventListener('click', handleAuth);
    elements.apiKeyInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAuth();
    });
    
    // Logout
    elements.logoutBtn.addEventListener('click', logout);
    
    // Theme toggle
    elements.themeToggleBtn.addEventListener('click', cycleTheme);
    
    // Navigation
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            navigateTo(page);
        });
    });
    
    // Refresh
    elements.refreshBtn.addEventListener('click', loadAllData);
    
    // Providers
    elements.addProviderBtn.addEventListener('click', () => {
        state.editingProvider = null;
        elements.providerName.value = '';
        elements.providerName.disabled = false;
        elements.providerDisplayName.value = '';
        elements.providerBaseUrl.value = '';
        elements.providerFormError.classList.add('hidden');
        document.querySelector('#provider-modal .modal-header h3').textContent = 'Add Custom Provider';
        openModal('provider-modal');
    });
    elements.saveProviderBtn.addEventListener('click', saveProvider);
    
    // Models
    elements.addModelBtn.addEventListener('click', () => openModelModal());
    elements.addRoutingBtn.addEventListener('click', addRoutingEntry);
    elements.saveModelBtn.addEventListener('click', saveModel);
    
    // Import/Export
    elements.exportBtn.addEventListener('click', exportConfig);
    elements.importFile.addEventListener('change', handleImportFileSelect);
    elements.importBtn.addEventListener('click', importConfig);
    
    // Validation
    elements.runValidationBtn.addEventListener('click', runValidation);
    
    // Modal closes
    document.querySelectorAll('.modal-close, .modal-cancel').forEach(btn => {
        btn.addEventListener('click', closeAllModals);
    });
    
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', closeAllModals);
    });
}

// API Helper
async function apiCall(endpoint, options = {}) {
    const url = `/setup/api${endpoint}`;
    const config = {
        headers: {
            'Authorization': `Bearer ${state.apiKey}`,
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };
    
    if (config.body && typeof config.body === 'object') {
        config.body = JSON.stringify(config.body);
    }
    
    try {
        const response = await fetch(url, config);
        
        if (response.status === 401) {
            showAuthError('Invalid API key');
            logout();
            return null;
        }
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Authentication
async function handleAuth() {
    const key = elements.apiKeyInput.value.trim();
    if (!key) {
        showAuthError('Please enter your API key');
        return;
    }
    
    state.apiKey = key;
    await verifyAndLoad();
}

async function verifyAndLoad() {
    try {
        const result = await apiCall('/status');
        if (result) {
            localStorage.setItem('modelProxyApiKey', state.apiKey);
            showApp();
            await loadAllData();
        }
    } catch (error) {
        showAuthError('Authentication failed. Please check your API key.');
        state.apiKey = '';
    }
}

function showAuthError(message) {
    elements.authError.querySelector('span').textContent = message;
    elements.authError.classList.remove('hidden');
}

function showApp() {
    elements.authScreen.classList.add('hidden');
    elements.app.classList.remove('hidden');
}

function logout() {
    localStorage.removeItem('modelProxyApiKey');
    state.apiKey = '';
    elements.apiKeyInput.value = '';
    elements.authError.classList.add('hidden');
    elements.app.classList.add('hidden');
    elements.authScreen.classList.remove('hidden');
}

// Navigation
function navigateTo(page) {
    state.currentPage = page;
    
    // Update nav items
    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });
    
    // Update page visibility
    elements.pages.forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });
    
    // Update page title
    const titles = {
        'dashboard': 'Dashboard',
        'providers': 'LLM Providers',
        'models': 'Model Routing',
        'apikeys': 'API Keys',
        'import-export': 'Import & Export',
        'validate': 'Validate Configuration'
    };
    elements.pageTitle.textContent = titles[page] || 'Configuration';
    
    // Refresh data if needed
    if (page === 'providers') renderProviders();
    if (page === 'models') renderModels();
    if (page === 'apikeys') loadAndRenderApiKeys();
}

// Data Loading
async function loadAllData() {
    showToast('Loading configuration...', 'info');
    
    try {
        // Load status
        const status = await apiCall('/status');
        if (status) {
            updateDashboardStats(status.stats);
            elements.dashConfigDir.textContent = status.config_dir;
            elements.dashLastUpdated.textContent = new Date().toLocaleString();
        }
        
        // Load providers
        const providersResult = await apiCall('/providers');
        if (providersResult) {
            state.providers = Object.entries(providersResult.providers).map(([name, config]) => ({
                ...config,
                isPreconfigured: isPreconfiguredProvider(name)
            }));
            updateNavCounts();
            if (state.currentPage === 'providers') renderProviders();
            if (state.currentPage === 'apikeys') loadAndRenderApiKeys();
        }
        
        // Load models
        const modelsResult = await apiCall('/models');
        if (modelsResult) {
            state.models = modelsResult.models;
            updateNavCounts();
            if (state.currentPage === 'models') renderModels();
        }
        
        // Load templates
        const templatesResult = await apiCall('/templates');
        if (templatesResult) {
            state.templates = templatesResult.templates;
        }
        
        showToast('Configuration loaded', 'success');
    } catch (error) {
        showToast('Failed to load configuration', 'error');
    }
}

function isPreconfiguredProvider(name) {
    const preconfigured = [
        'anthropic', 'azure', 'cerebras', 'chutes', 'cloudflare',
        'gemini', 'github', 'groq', 'llama', 'longcat',
        'mistral', 'nahcrof', 'openai', 'openrouter', 'zai'
    ];
    return preconfigured.includes(name.toLowerCase());
}

function updateNavCounts() {
    document.getElementById('nav-provider-count').textContent = state.providers.length;
    document.getElementById('nav-model-count').textContent = state.models.length;
}

function updateDashboardStats(stats) {
    elements.dashProviderCount.textContent = stats.total_providers;
    elements.dashEnabledProviders.textContent = `${stats.enabled_providers} enabled`;
    elements.dashModelCount.textContent = stats.total_models;
    elements.dashKeysCount.textContent = '?';
    elements.dashKeysProviders.textContent = `${stats.providers_with_keys} providers`;
}

// Render Providers
function renderProviders() {
    const preconfigured = state.providers.filter(p => p.isPreconfigured);
    const custom = state.providers.filter(p => !p.isPreconfigured);
    
    elements.preconfiguredCount.textContent = `${preconfigured.length} providers`;
    elements.customCount.textContent = `${custom.length} providers`;
    
    // Preconfigured providers grid
    elements.preconfiguredProviders.innerHTML = preconfigured.map(provider => {
        const info = PROVIDER_INFO[provider.name.toLowerCase()] || { icon: 'fa-cloud', color: '#6b7280' };
        return `
            <div class="provider-card ${provider.enabled ? 'enabled' : ''}">
                <div class="provider-card-header">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <i class="fas ${info.icon}" style="color: ${info.color}; font-size: 20px;"></i>
                        <h4>${provider.display_name}</h4>
                    </div>
                    <div class="provider-card-actions">
                        <button class="btn-icon" onclick="event.stopPropagation(); editProvider('${provider.name}')" title="Edit provider">
                            <i class="fas fa-edit"></i>
                        </button>
                        <div class="provider-status ${provider.enabled ? 'enabled' : ''}" onclick="event.stopPropagation(); toggleProvider('${provider.name}')" title="Click to ${provider.enabled ? 'disable' : 'enable'}"></div>
                    </div>
                </div>
                <p>${provider.endpoints?.base_url || 'No URL configured'}</p>
                <div class="provider-meta">
                    <span class="provider-format">${provider.endpoints?.compatible_format || 'unknown'}</span>
                    <span class="provider-status-text">${provider.enabled ? 'Enabled' : 'Disabled'}</span>
                </div>
            </div>
        `;
    }).join('');
    
    // Custom providers list
    elements.customProvidersList.innerHTML = custom.map(provider => `
        <div class="provider-item ${provider.enabled ? 'enabled' : ''}">
            <div class="provider-item-info" style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div class="provider-status ${provider.enabled ? 'enabled' : ''}" onclick="event.stopPropagation(); toggleProvider('${provider.name}')" style="cursor: pointer; width: 10px; height: 10px; border-radius: 50%;" title="Click to ${provider.enabled ? 'disable' : 'enable'}"></div>
                    <h4>${provider.display_name}</h4>
                </div>
                <p>${provider.endpoints?.base_url || 'No URL configured'}</p>
            </div>
            <div class="provider-item-actions">
                <button class="btn btn-text btn-sm" onclick="event.stopPropagation(); editProvider('${provider.name}')" title="Edit provider">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-text btn-sm" onclick="event.stopPropagation(); deleteProvider('${provider.name}')" title="Delete provider">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
}

async function toggleProvider(providerName) {
    const provider = state.providers.find(p => p.name === providerName);
    if (!provider) {
        showToast('Provider not found', 'error');
        return;
    }
    
    const newState = !provider.enabled;
    
    try {
        const updatedProvider = { ...provider, enabled: newState };
        await apiCall('/providers', {
            method: 'POST',
            body: updatedProvider
        });
        provider.enabled = newState;
        renderProviders();
        showToast(`${provider.display_name} ${newState ? 'enabled' : 'disabled'}`, 'success');
    } catch (error) {
        showToast(`Failed to ${newState ? 'enable' : 'disable'} ${provider.display_name}`, 'error');
    }
}

async function deleteProvider(providerName) {
    if (!confirm(`Delete provider "${providerName}"?`)) return;
    
    try {
        await apiCall(`/providers/${providerName}`, { method: 'DELETE' });
        state.providers = state.providers.filter(p => p.name !== providerName);
        renderProviders();
        updateNavCounts();
        showToast('Provider deleted', 'success');
    } catch (error) {
        showToast('Failed to delete provider', 'error');
    }
}

// Edit Provider
function editProvider(providerName) {
    const provider = state.providers.find(p => p.name === providerName);
    if (!provider) {
        showToast('Provider not found', 'error');
        return;
    }
    
    state.editingProvider = providerName;
    
    // Populate form fields
    elements.providerName.value = provider.name;
    elements.providerName.disabled = true;
    elements.providerDisplayName.value = provider.display_name;
    elements.providerBaseUrl.value = provider.endpoints?.base_url || '';
    elements.providerFormat.value = provider.endpoints?.compatible_format || 'openai';
    
    // Update modal title
    document.querySelector('#provider-modal .modal-header h3').textContent = 'Edit Provider';
    
    // Clear any errors
    elements.providerFormError.classList.add('hidden');
    
    openModal('provider-modal');
}

// Modals
function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeAllModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    
    // Reset provider editing state
    if (state.editingProvider) {
        state.editingProvider = null;
        elements.providerName.value = '';
        elements.providerName.disabled = false;
        elements.providerDisplayName.value = '';
        elements.providerBaseUrl.value = '';
        document.querySelector('#provider-modal .modal-header h3').textContent = 'Add Custom Provider';
    }
}

// Provider Modal
async function saveProvider() {
    const isEditing = state.editingProvider !== null;
    const name = elements.providerName.value.trim().toLowerCase();
    const displayName = elements.providerDisplayName.value.trim();
    const baseUrl = elements.providerBaseUrl.value.trim();
    const format = elements.providerFormat.value;
    
    if (!name || !displayName || !baseUrl) {
        showFormError('provider-form-error', 'All fields are required');
        return;
    }
    
    let existingProvider = null;
    if (isEditing) {
        existingProvider = state.providers.find(p => p.name === name);
    }
    
    const template = state.templates[format] || {};
    
    const provider = {
        name,
        display_name: displayName,
        enabled: existingProvider ? existingProvider.enabled : true,
        api_keys: existingProvider ? existingProvider.api_keys : (template.api_keys || { env_var_patterns: [`${name.toUpperCase()}_API_KEY`, `${name.toUpperCase()}_API_KEY_{INDEX}`] }),
        endpoints: {
            ...(existingProvider ? existingProvider.endpoints : template.endpoints),
            base_url: baseUrl,
            compatible_format: format
        },
        authentication: existingProvider ? existingProvider.authentication : (template.authentication || { type: 'bearer', header_name: 'Authorization', header_format: 'Bearer {api_key}' }),
        request_config: existingProvider ? existingProvider.request_config : (template.request_config || { timeout_seconds: 60, max_retries: 3, retry_on_status: [429, 500, 502, 503, 504] }),
        rate_limiting: existingProvider ? existingProvider.rate_limiting : (template.rate_limiting || { enabled: false, cooldown_seconds: 180 }),
        error_handling: existingProvider ? existingProvider.error_handling : (template.error_handling || { '401': { action: 'global_key_failure' }, '429': { action: 'model_key_failure' } })
    };
    
    try {
        await apiCall('/providers?overwrite=true', {
            method: 'POST',
            body: provider
        });
        
        if (isEditing) {
            const index = state.providers.findIndex(p => p.name === name);
            if (index !== -1) {
                state.providers[index] = { ...provider, isPreconfigured: state.providers[index].isPreconfigured };
            }
            showToast('Provider updated successfully', 'success');
        } else {
            state.providers.push({ ...provider, isPreconfigured: false });
            showToast('Provider added successfully', 'success');
        }
        
        closeAllModals();
        renderProviders();
        updateNavCounts();
        
        elements.providerName.value = '';
        elements.providerName.disabled = false;
        elements.providerDisplayName.value = '';
        elements.providerBaseUrl.value = '';
        state.editingProvider = null;
        document.querySelector('#provider-modal .modal-header h3').textContent = 'Add Custom Provider';
    } catch (error) {
        showFormError('provider-form-error', error.message);
    }
}

// Render Models
function renderModels() {
    elements.modelsList.innerHTML = state.models.map(model => `
        <div class="model-card">
            <div class="model-card-header">
                <div class="model-card-title">
                    <h3>${model.config.logical_name}</h3>
                </div>
                <div class="model-card-badges">
                    <span class="badge-outline">Timeout: ${model.config.timeout_seconds}s</span>
                    <span class="badge-outline">Cooldown: ${model.config.default_cooldown_seconds}s</span>
                    <button class="btn btn-text btn-sm" onclick="editModel('${model.name}')">
                        <i class="fas fa-edit"></i> Edit
                    </button>
                    <button class="btn btn-text btn-sm" onclick="deleteModel('${model.name}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="model-routing-chain">
                ${model.config.model_routings.map((routing, idx) => `
                    <div class="routing-step">
                        ${idx > 0 ? '<i class="fas fa-arrow-right routing-arrow"></i>' : ''}
                        <div class="routing-badge">
                            <span class="provider-name">${routing.provider}</span>
                            <span class="model-name">${routing.model}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

function openModelModal(modelName = null) {
    state.editingModel = modelName;
    
    if (modelName) {
        const model = state.models.find(m => m.name === modelName);
        if (!model) return;
        
        elements.modelModalTitle.textContent = 'Edit Model Routing';
        elements.modelLogicalName.value = model.config.logical_name;
        elements.modelLogicalName.disabled = true;
        elements.modelTimeout.value = model.config.timeout_seconds;
        elements.modelCooldown.value = model.config.default_cooldown_seconds;
        
        elements.modelRoutingsList.innerHTML = '';
        model.config.model_routings.forEach(routing => addRoutingEntry(routing.provider, routing.model));
    } else {
        elements.modelModalTitle.textContent = 'Add Model Routing';
        elements.modelLogicalName.value = '';
        elements.modelLogicalName.disabled = false;
        elements.modelTimeout.value = '180';
        elements.modelCooldown.value = '180';
        elements.modelRoutingsList.innerHTML = '';
        addRoutingEntry();
    }
    
    elements.modelFormError.classList.add('hidden');
    openModal('model-modal');
}

function addRoutingEntry(provider = '', model = '') {
    const entry = document.createElement('div');
    entry.className = 'routing-entry';
    entry.innerHTML = `
        <select class="routing-provider">
            <option value="">Select provider...</option>
            ${state.providers.filter(p => p.enabled).map(p => `
                <option value="${p.name}" ${p.name === provider ? 'selected' : ''}>${p.display_name}</option>
            `).join('')}
        </select>
        <input type="text" class="routing-model" placeholder="Model ID (e.g., gpt-4)" value="${model}">
        <button type="button" class="btn btn-text btn-sm" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    elements.modelRoutingsList.appendChild(entry);
}

async function saveModel() {
    const logicalName = elements.modelLogicalName.value.trim();
    const timeout = parseInt(elements.modelTimeout.value) || 180;
    const cooldown = parseInt(elements.modelCooldown.value) || 180;
    
    if (!logicalName) {
        showFormError('model-form-error', 'Logical name is required');
        return;
    }
    
    const routings = [];
    elements.modelRoutingsList.querySelectorAll('.routing-entry').forEach(entry => {
        const provider = entry.querySelector('.routing-provider').value;
        const model = entry.querySelector('.routing-model').value.trim();
        if (provider && model) {
            routings.push({ provider, model });
        }
    });
    
    if (routings.length === 0) {
        showFormError('model-form-error', 'At least one routing is required');
        return;
    }
    
    const model = {
        logical_name: logicalName,
        timeout_seconds: timeout,
        default_cooldown_seconds: cooldown,
        model_routings: routings,
        fallback_model_routings: []
    };
    
    try {
        await apiCall('/models', {
            method: 'POST',
            body: model,
            overwrite: state.editingModel !== null
        });
        
        closeAllModals();
        await loadAllData();
        showToast('Model saved successfully', 'success');
    } catch (error) {
        showFormError('model-form-error', error.message);
    }
}

function editModel(modelName) {
    openModelModal(modelName);
}

async function deleteModel(modelName) {
    if (!confirm(`Delete model "${modelName}"?`)) return;
    
    try {
        await apiCall(`/models/${modelName}`, { method: 'DELETE' });
        state.models = state.models.filter(m => m.name !== modelName);
        renderModels();
        updateNavCounts();
        showToast('Model deleted', 'success');
    } catch (error) {
        showToast('Failed to delete model', 'error');
    }
}

// Load and render API Keys
async function loadAndRenderApiKeys() {
    const enabledProviders = state.providers.filter(p => p.enabled);
    
    const providerKeys = {};
    for (const provider of enabledProviders) {
        try {
            const status = await apiCall(`/providers/${provider.name}/keys`);
            if (status) {
                providerKeys[provider.name] = status;
            }
        } catch (error) {
            // Silently ignore - provider may not have keys endpoint
        }
    }
    
    renderApiKeys(providerKeys);
}

function renderApiKeys(providerKeys = {}) {
    const enabledProviders = state.providers.filter(p => p.enabled);
    
    if (enabledProviders.length === 0) {
        elements.apiKeysContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-key"></i>
                <h3>No Enabled Providers</h3>
                <p>Enable at least one provider in the Providers section to add API keys.</p>
                <button class="btn btn-primary" onclick="navigateTo('providers')">
                    Go to Providers
                </button>
            </div>
        `;
        return;
    }
    
    elements.apiKeysContainer.innerHTML = enabledProviders.map(provider => {
        const keyStatus = providerKeys[provider.name] || { has_keys: false, key_count: 0, key_preview: [] };
        const hasKeys = keyStatus.has_keys;
        
        return `
            <div class="api-key-card" data-provider="${provider.name}">
                <div class="api-key-card-header">
                    <div class="api-key-title">
                        <i class="fas fa-cloud" style="color: var(--primary-500);"></i>
                        <h3>${provider.display_name}</h3>
                    </div>
                    <div class="api-key-status">
                        <span class="status-dot ${hasKeys ? 'ok' : 'missing'}"></span>
                        <span>${hasKeys ? `${keyStatus.key_count} key(s) configured` : 'No keys configured'}</span>
                    </div>
                </div>
                
                ${hasKeys ? `
                    <div class="existing-keys">
                        <div class="existing-keys-header">
                            <span>Existing Keys</span>
                            <button class="btn btn-text btn-sm" onclick="toggleKeyForm('${provider.name}')">
                                <i class="fas fa-plus"></i> Add Another Key
                            </button>
                        </div>
                        ${keyStatus.key_preview.map((preview, idx) => `
                            <div class="key-item">
                                <div class="key-info">
                                    <i class="fas fa-key"></i>
                                    <code class="key-preview">${preview}</code>
                                </div>
                                <div class="key-actions">
                                    <button class="btn-icon" onclick="editApiKey('${provider.name}', ${idx})" title="Edit key">
                                        <i class="fas fa-edit"></i>
                                    </button>
                                    <button class="btn-icon btn-danger" onclick="deleteApiKey('${provider.name}', ${idx})" title="Delete key">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="api-key-form ${hasKeys ? 'hidden' : ''}" id="key-form-${provider.name}">
                    <div class="form-section-title">${hasKeys ? 'Add New Key' : 'Add API Key'}</div>
                    <div class="key-input-row">
                        <input type="text" class="key-env-var" id="key-env-${provider.name}" value="${provider.name.toUpperCase()}_API_KEY_${hasKeys ? keyStatus.key_count + 1 : 1}" placeholder="Environment variable name">
                        <input type="password" class="key-value" id="key-val-${provider.name}" placeholder="Enter API key">
                        <button class="btn btn-primary" onclick="saveApiKey('${provider.name}')">
                            <i class="fas fa-save"></i> Save
                        </button>
                    </div>
                </div>
                
                ${hasKeys ? `
                    <div class="key-note">
                        <i class="fas fa-info-circle"></i>
                        <strong>Important:</strong> To edit or delete keys, you must manually update your <code>.env</code> file. 
                        Click the edit button to see instructions.
                    </div>
                ` : `
                    <div class="key-note">
                        <i class="fas fa-info-circle"></i>
                        API keys are stored in environment variables (your <code>.env</code> file), not in configuration files.
                    </div>
                `}
            </div>
        `;
    }).join('');
}

function toggleKeyForm(providerName) {
    const form = document.getElementById(`key-form-${providerName}`);
    form.classList.toggle('hidden');
    if (!form.classList.contains('hidden')) {
        document.getElementById(`key-env-${providerName}`).focus();
    }
}

async function saveApiKey(providerName) {
    const envVar = document.getElementById(`key-env-${providerName}`).value.trim();
    const keyValue = document.getElementById(`key-val-${providerName}`).value.trim();
    
    if (!envVar || !keyValue) {
        showToast('Please enter both environment variable name and API key', 'error');
        return;
    }
    
    try {
        await navigator.clipboard.writeText(`${envVar}=${keyValue}`);
        showToast('API key copied to clipboard! Add it to your .env file.', 'success');
    } catch (err) {
        showToast(`Add to .env: ${envVar}=${keyValue}`, 'info', 10000);
    }
    
    document.getElementById(`key-val-${providerName}`).value = '';
}

function editApiKey(providerName, keyIndex) {
    const provider = state.providers.find(p => p.name === providerName);
    if (!provider) return;
    
    const message = `
        To edit an API key for ${provider.display_name}:
        
        1. Open your .env file
        2. Find the environment variable for this key
        3. Update the value
        4. Restart the server
        
        Common variable names:
        - ${provider.name.toUpperCase()}_API_KEY
        - ${provider.name.toUpperCase()}_API_KEY_1
        - ${provider.name.toUpperCase()}_API_KEY_2
        
        Note: For security, actual key values cannot be retrieved or edited through this interface.
    `;
    
    showToast(message, 'info', 15000);
}

function deleteApiKey(providerName, keyIndex) {
    const provider = state.providers.find(p => p.name === providerName);
    if (!provider) return;
    
    if (!confirm(`Are you sure you want to delete this API key for ${provider.display_name}?\n\nNote: This only removes the reference. You must also manually remove it from your .env file.`)) {
        return;
    }
    
    const message = `
        To fully delete this API key:
        
        1. Open your .env file
        2. Remove the line containing the key
        3. Restart the server
        
        The key reference has been marked for removal but requires manual .env file update.
    `;
    
    showToast(message, 'info', 15000);
}

// Export/Import
async function exportConfig() {
    try {
        const data = await apiCall('/export', {
            method: 'POST',
            body: { include_env: true }
        });
        elements.exportPreview.textContent = JSON.stringify(data, null, 2);
        elements.exportPreview.classList.remove('hidden');
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `model-proxy-config-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showToast('Configuration exported and downloaded', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast(`Export failed: ${error.message || 'Unknown error'}`, 'error');
    }
}

function handleImportFileSelect(e) {
    const file = e.target.files[0];
    if (!file) {
        elements.importFileName.textContent = 'No file selected';
        elements.importBtn.disabled = true;
        return;
    }
    
    elements.importFileName.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const data = JSON.parse(event.target.result);
            state.importData = data;
            elements.importPreview.textContent = JSON.stringify(data, null, 2);
            elements.importPreview.classList.remove('hidden');
            elements.importBtn.disabled = false;
        } catch (error) {
            elements.importPreview.textContent = `Error: Invalid JSON - ${error.message}`;
            elements.importPreview.classList.remove('hidden');
            elements.importBtn.disabled = true;
        }
    };
    reader.readAsText(file);
}

async function importConfig() {
    if (!state.importData) return;
    
    const merge = elements.importMerge.checked;
    
    try {
        const result = await apiCall('/import?merge=' + merge, {
            method: 'POST',
            body: state.importData
        });
        
        // Show success message
        showToast(result.note, 'success');
        
        // Download the complete .env file with all API keys
        if (result.env_file) {
            const envBlob = new Blob([result.env_file], { type: 'text/plain' });
            const envUrl = URL.createObjectURL(envBlob);
            const a = document.createElement('a');
            a.href = envUrl;
            a.download = result.env_filename || '.env';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(envUrl);
            
            showToast('Complete .env file downloaded with all API keys! Place it in your project root and restart the server.', 'info', 15000);
        }
        
        await loadAllData();
        
        elements.importFile.value = '';
        elements.importFileName.textContent = 'No file selected';
        elements.importPreview.classList.add('hidden');
        elements.importBtn.disabled = true;
        state.importData = null;
    } catch (error) {
        showToast(`Import failed: ${error.message}`, 'error');
    }
}

// Validation
async function runValidation() {
    try {
        elements.runValidationBtn.disabled = true;
        elements.runValidationBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
        
        const result = await apiCall('/validate');
        
        if (result.valid) {
            elements.validationResults.innerHTML = `
                <div class="validation-list">
                    <div class="validation-item success">
                        <i class="fas fa-check-circle" style="font-size: 24px;"></i>
                        <div>
                            <div style="font-weight: 700;">All configurations are valid!</div>
                            <div style="font-size: 13px; margin-top: 4px;">${result.message}</div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            elements.validationResults.innerHTML = `
                <div class="validation-list">
                    ${result.errors.map(err => `
                        <div class="validation-item error">
                            <i class="fas fa-exclamation-circle" style="font-size: 24px;"></i>
                            <div>${err}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        showToast(result.valid ? 'Validation passed' : 'Validation failed', result.valid ? 'success' : 'error');
    } catch (error) {
        showToast('Validation error: ' + error.message, 'error');
    } finally {
        elements.runValidationBtn.disabled = false;
        elements.runValidationBtn.innerHTML = '<i class="fas fa-play"></i> Run Validation';
    }
}

// Toast Notifications
function showToast(message, type = 'info', duration = 5000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type]} toast-icon"></i>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Form Helpers
function showFormError(elementId, message) {
    const errorEl = document.getElementById(elementId);
    errorEl.querySelector('span').textContent = message;
    errorEl.classList.remove('hidden');
}

// Make functions globally accessible for onclick handlers
window.navigateTo = navigateTo;
window.toggleProvider = toggleProvider;
window.deleteProvider = deleteProvider;
window.editProvider = editProvider;
window.editModel = editModel;
window.deleteModel = deleteModel;
window.saveApiKey = saveApiKey;
window.editApiKey = editApiKey;
window.deleteApiKey = deleteApiKey;
window.toggleKeyForm = toggleKeyForm;
