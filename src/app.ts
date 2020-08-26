import {
  drawDatasetThumbnails,
  generateData,
  hideControls,
  initTutorial,
  makeGUI,
  reset
} from './playground';
import './styles.css';

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
