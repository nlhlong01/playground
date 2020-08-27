import {
  drawDatasetThumbnails,
  generateData,
  hideControls,
  initTutorial,
  makeGUI,
  reset
} from './playground';
import './styles.css';
import 'material-design-lite';
import 'material-design-lite/dist/material.indigo-blue.min.css';
import './analytics';

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
