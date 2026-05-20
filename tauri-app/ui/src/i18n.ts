import { addMessages, init, getLocaleFromNavigator } from 'svelte-i18n';
import en from './locales/en.json';
import hi from './locales/hi.json';

addMessages('en', en);
addMessages('hi', hi);

init({
  fallbackLocale: 'en',
  initialLocale: getLocaleFromNavigator() || 'en',
});